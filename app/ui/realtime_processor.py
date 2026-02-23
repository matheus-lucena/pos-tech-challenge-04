import os
import queue
import threading
import time
import wave
from datetime import datetime
from typing import Optional

import librosa
import numpy as np
import pyaudio
import traceback

from config.constants import (
    AUDIO_CHUNK_BYTES_FILE,
    AUDIO_CHUNK_SIZE,
    INT16_MAX,
    INT16_NORMALIZE,
    SAMPLE_RATE,
    STREAM_WARMUP_SEC,
    TEMP_AUDIO_DIR,
    VIOLENCE_ALERT_PREFIX,
    ERROR_PREFIX,
    WAV_CHANNELS,
    WAV_SAMPLEWIDTH,
    WAVEFORM_DOWNSAMPLE_FACTOR,
)
from services.instances import get_s3_service, get_transcribe_streaming_service


class RealtimeAudioProcessor:
    def __init__(self):
        self.streaming_service = get_transcribe_streaming_service()
        self.s3_service = get_s3_service()
        self.is_processing = False
        self.current_transcript = ""
        self.transcript_parts = []
        self.audio_stream = None
        self.pyaudio_instance = None
        self.transcript_thread = None
        self.recorded_audio_frames = []
        self.audio_file_path = None
        self.violence_alert_message = None
    
    def start_realtime_transcription(
        self,
        audio_file_path: Optional[str] = None,
        language_code: str = "pt-BR"
    ) -> str:
        if self.is_processing:
            return "⚠️ A transcription is already in progress. Stop the previous one first."
        
        self.is_processing = True
        self.current_transcript = ""
        self.transcript_parts = []
        
        try:
            if audio_file_path:
                return self._process_audio_file(audio_file_path, language_code)
            self.start_microphone_streaming(language_code)
            return "✅ Live transcription started."
        except Exception as e:
            self.is_processing = False
            return f"❌ Error starting transcription: {str(e)}"
    
    def _process_audio_file(
        self,
        audio_file_path: str,
        language_code: str
    ) -> str:
        try:
            y, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, mono=True)
            audio_int16 = (y * INT16_MAX).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            chunks = [
                audio_bytes[i : i + AUDIO_CHUNK_BYTES_FILE]
                for i in range(0, len(audio_bytes), AUDIO_CHUNK_BYTES_FILE)
            ]
            
            transcript_gen = self.streaming_service.start_stream(language_code)
            
            def send_chunks():
                for chunk in chunks:
                    if not self.is_processing:
                        break
                    self.streaming_service.send_audio_chunk(chunk)
                    time.sleep(0.01)
                self.streaming_service.stop_stream()
            
            send_thread = threading.Thread(target=send_chunks, daemon=True)
            send_thread.start()
            
            full_transcript = ""
            for transcript, is_final in transcript_gen:
                if not self.is_processing:
                    break
                if transcript:
                    if is_final:
                        full_transcript += transcript + " "
                        self.transcript_parts.append(transcript)
                    else:
                        current_partial = transcript
            
            self.current_transcript = full_transcript.strip()
            self.is_processing = False
            
            return f"✅ Transcription completed: {len(self.transcript_parts)} segments processed."
            
        except Exception as e:
            self.is_processing = False
            return f"❌ Error processing file: {str(e)}"
    
    def start_microphone_streaming(self, language_code: str = "pt-BR", device_index: Optional[int] = None):
        if self.is_processing:
            raise RuntimeError("A transcription is already in progress.")
        
        self.is_processing = True
        self.current_transcript = ""
        self.transcript_parts = []
        self.last_partial_transcript = ""
        self.recorded_audio_frames = []
        self.violence_alert_message = None

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            if device_index is not None:
                self._log_device_info(device_index)
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=WAV_CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK_SIZE,
                input_device_index=device_index,
            )
            print(
                f"✅ Audio stream opened: {WAV_CHANNELS} ch, {SAMPLE_RATE}Hz, {AUDIO_CHUNK_SIZE} frames",
                flush=True,
            )
            self.streaming_service.start_stream(language_code)
            time.sleep(STREAM_WARMUP_SEC)

            def send_audio():
                self._run_send_audio_loop()

            def receive_transcripts():
                try:
                    print("Starting transcript reception...", flush=True)
                    first_result = True
                    while self.is_processing:
                        try:
                            result = self.streaming_service.result_queue.get(timeout=0.1)
                            
                            if not isinstance(result, tuple) or len(result) != 3:
                                print(f"⚠️ Unexpected result format: {result}", flush=True)
                                continue
                            
                            transcript, is_final, is_violent = result
                            
                            if isinstance(transcript, str) and transcript.startswith(ERROR_PREFIX):
                                print(f"❌ Error received: {transcript}", flush=True)
                                break
                            
                            if isinstance(transcript, str) and transcript.startswith(VIOLENCE_ALERT_PREFIX):
                                self._set_violence_alert(transcript)
                                continue
                            if first_result and transcript and transcript.strip():
                                print("📥 First transcript result received.", flush=True)
                                first_result = False
                            if transcript and transcript.strip():
                                self._apply_transcript_result(transcript.strip(), is_final, is_violent)
                            elif transcript:
                                print("Empty transcript received")
                        except queue.Empty:
                            if not self.streaming_service.is_streaming:
                                print("⚠️ Streaming stopped, exiting receive loop", flush=True)
                                break
                            continue
                        except ValueError as e:
                            print(f"❌ Error unpacking result: {e}")
                            print(f"   Result received: {result if 'result' in locals() else 'N/A'}")
                            traceback.print_exc()
                            continue
                        except Exception as e:
                            if self.is_processing:
                                print(f"❌ Error receiving transcripts: {e}")
                                traceback.print_exc()
                            break
                except Exception as e:
                    if self.is_processing:
                        print(f"Error receiving transcripts: {e}")
                        traceback.print_exc()
            
            send_thread = threading.Thread(target=send_audio, daemon=True)
            self.transcript_thread = threading.Thread(target=receive_transcripts, daemon=True)
            self.transcript_thread.start()
            send_thread.start()
            
        except Exception as e:
            self.is_processing = False
            self._cleanup_audio()
            raise RuntimeError(f"Error starting capture: {str(e)}")

    def _log_device_info(self, device_index: int):
        try:
            device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
            print(f"🎤 Using device: {device_info['name']}")
            print(f"   Supported sample rate: {device_info.get('defaultSampleRate', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Warning checking device: {e}")

    def _run_send_audio_loop(self):
        chunks_sent = 0
        print(f"🎤 Starting audio send (CHUNK: {AUDIO_CHUNK_SIZE} bytes)", flush=True)
        try:
            while self.is_processing:
                try:
                    if not self.streaming_service.is_streaming:
                        print("⚠️ Streaming stopped, interrupting send", flush=True)
                        break
                    audio_data = self.audio_stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                    if len(audio_data) == 0:
                        print("⚠️ Empty audio chunk received")
                        continue
                    if len(audio_data) % 2 != 0:
                        print(f"⚠️ Invalid chunk size: {len(audio_data)} (must be even)")
                        continue
                    self.recorded_audio_frames.append(audio_data)
                    if self.is_processing and self.streaming_service.is_streaming:
                        self.streaming_service.send_audio_chunk(audio_data)
                        chunks_sent += 1
                    else:
                        print("⚠️ Streaming not active, stopping send", flush=True)
                        break
                except Exception as e:
                    if self.is_processing:
                        print(f"❌ Error sending audio: {e}", flush=True)
                        traceback.print_exc()
                    break
            print(f"✅ Audio send finished. Total chunks: {chunks_sent}", flush=True)
        finally:
            if self.streaming_service.is_streaming:
                self.streaming_service.stop_stream()

    def _set_violence_alert(self, transcript: str):
        payload = transcript.replace(VIOLENCE_ALERT_PREFIX, "").strip()
        parts = payload.split("|", 1)
        label = parts[0] if parts else "Risk detected"
        score = parts[1] if len(parts) > 1 else ""
        self.violence_alert_message = f"{label} ({score})" if score else label

    def _apply_transcript_result(self, transcript_clean: str, is_final: bool, is_violent: bool):
        violence_indicator = " ⚠️ VIOLENCE RISK DETECTED!" if is_violent else ""
        print(
            f"Transcript received: '{transcript_clean}' (final: {is_final}, risk: {is_violent}){violence_indicator}"
        )
        if is_final:
            if transcript_clean and transcript_clean not in [p.strip() for p in self.transcript_parts]:
                self.transcript_parts.append(transcript_clean)
            self.last_partial_transcript = ""
            self.current_transcript = " ".join(self.transcript_parts).strip()
            print(f"Transcript updated (final): {self.current_transcript}")
        else:
            self.last_partial_transcript = transcript_clean
            base_text = " ".join(self.transcript_parts).strip()
            self.current_transcript = (base_text + " " + self.last_partial_transcript) if base_text else self.last_partial_transcript
            print(f"Transcript updated (partial): {self.current_transcript}")

    def _cleanup_audio(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None

        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None
    
    def stop_transcription(self) -> str:
        if not self.is_processing:
            return "⚠️ No transcription in progress."
        
        self.is_processing = False
        self.streaming_service.stop_stream()
        self._save_recorded_audio()
        self._cleanup_audio()
        
        if self.transcript_thread:
            self.transcript_thread.join(timeout=2)
        
        return "⏹️ Transcription stopped."
    
    def _save_recorded_audio(self):
        if not self.recorded_audio_frames:
            return None
        try:
            os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(TEMP_AUDIO_DIR, filename)
            wf = wave.open(filepath, "wb")
            wf.setnchannels(WAV_CHANNELS)
            wf.setsampwidth(WAV_SAMPLEWIDTH)
            wf.setframerate(SAMPLE_RATE)
            
            for frame in self.recorded_audio_frames:
                wf.writeframes(frame)
            
            wf.close()
            self.audio_file_path = filepath
            return filepath
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None
    
    def get_recorded_audio_path(self) -> Optional[str]:
        return self.audio_file_path
    
    def get_audio_waveform_data(self):
        if not self.recorded_audio_frames:
            return None
        try:
            audio_bytes = b"".join(self.recorded_audio_frames)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / INT16_NORMALIZE
            downsampled = audio_float[::WAVEFORM_DOWNSAMPLE_FACTOR]
            times = np.arange(len(downsampled)) / (SAMPLE_RATE / WAVEFORM_DOWNSAMPLE_FACTOR)
            return {
                "times": times.tolist(),
                "values": downsampled.tolist(),
                "sample_rate": SAMPLE_RATE,
            }
        except Exception as e:
            print(f"Error processing waveform: {e}")
            return None
    
    def get_current_transcript(self) -> str:
        return self.current_transcript or "Waiting for transcription..."

    def get_violence_alert(self) -> Optional[str]:
        return self.violence_alert_message

    @staticmethod
    def list_audio_devices():
        devices = []
        try:
            p = pyaudio.PyAudio()
            preferred_apis = ("WASAPI", "Windows WASAPI", "MME", "Windows MME", "DirectSound", "Core Audio", "ALSA", "JACK", "PulseAudio")
            chosen_host_api = None
            try:
                api_count = p.get_host_api_count()
            except Exception:
                api_count = 0
            for a in range(api_count):
                try:
                    api_info = p.get_host_api_info_by_index(a)
                    api_name = (api_info.get("name") or "").strip()
                    for pref in preferred_apis:
                        if pref.lower() in api_name.lower():
                            chosen_host_api = a
                            break
                    if chosen_host_api is not None:
                        break
                except Exception:
                    continue
            device_count = p.get_device_count()
            seen = set()
            for i in range(device_count):
                info = p.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) <= 0:
                    continue
                if chosen_host_api is not None and info.get("hostApi") != chosen_host_api:
                    continue
                name = (info.get("name") or "").strip() or f"Device {i}"
                if chosen_host_api is None:
                    if name.lower() in seen:
                        continue
                    seen.add(name.lower())
                devices.append((i, name))
            p.terminate()
        except Exception as e:
            print(f"Error listing devices: {e}")
        return devices


_realtime_processor = RealtimeAudioProcessor()


def process_audio_realtime(audio_file: Optional[str]) -> tuple:
    if not audio_file:
        return "No audio provided.", "⚠️"
    
    try:
        status = _realtime_processor.start_realtime_transcription(audio_file)
        transcript = _realtime_processor.get_current_transcript()
        return transcript, status
    except Exception as e:
        return f"Error: {str(e)}", "❌"
