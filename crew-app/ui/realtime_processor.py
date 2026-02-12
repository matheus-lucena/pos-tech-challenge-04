import threading
import time
import pyaudio
import numpy as np
from typing import Optional
from services.transcribe_streaming_service import TranscribeStreamingService
from services.s3_service import S3Service


class RealtimeAudioProcessor:
    def __init__(self):
        self.streaming_service = TranscribeStreamingService()
        self.s3_service = S3Service()
        self.is_processing = False
        self.current_transcript = ""
        self.transcript_parts = []
        self.audio_stream = None
        self.pyaudio_instance = None
        self.transcript_thread = None
        self.recorded_audio_frames = []
        self.audio_file_path = None
    
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
            else:
                return self._start_microphone_stream(language_code)
        except Exception as e:
            self.is_processing = False
            return f"❌ Error starting transcription: {str(e)}"
    
    def _process_audio_file(
        self,
        audio_file_path: str,
        language_code: str
    ) -> str:
        import librosa
        import numpy as np
        
        try:
            y, sr = librosa.load(audio_file_path, sr=16000, mono=True)
            audio_int16 = (y * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            chunk_size = 1024 * 2
            chunks = [
                audio_bytes[i:i+chunk_size]
                for i in range(0, len(audio_bytes), chunk_size)
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
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            if device_index is not None:
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
                    print(f"🎤 Using device: {device_info['name']}")
                    print(f"   Supported sample rate: {device_info.get('defaultSampleRate', 'N/A')}")
                except Exception as e:
                    print(f"⚠️ Warning checking device: {e}")
            
            self.audio_stream = self.pyaudio_instance.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_index
            )
            
            print(f"✅ Audio stream opened: {CHANNELS} channel(s), {RATE}Hz, {FORMAT}, {CHUNK} frames per buffer")
            
            self.streaming_service.start_stream(language_code)
            time.sleep(0.5)
            
            def send_audio():
                try:
                    chunks_sent = 0
                    print(f"🎤 Starting audio send (CHUNK: {CHUNK} bytes)")
                    
                    while self.is_processing:
                        try:
                            if not self.streaming_service.is_streaming:
                                print("⚠️ Streaming stopped, interrupting send")
                                break
                            
                            audio_data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
                            
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
                                if chunks_sent % 20 == 0:
                                    print(f"📤 Sent {chunks_sent} audio chunks ({len(audio_data)} bytes each)")
                            else:
                                print("⚠️ Streaming not active, stopping send")
                                break
                        except Exception as e:
                            if self.is_processing:
                                print(f"❌ Error sending audio: {e}")
                                import traceback
                                traceback.print_exc()
                            break
                    print(f"✅ Audio send finished. Total chunks: {chunks_sent}")
                finally:
                    if self.streaming_service.is_streaming:
                        self.streaming_service.stop_stream()
            
            def receive_transcripts():
                try:
                    print("Starting transcript reception...")
                    while self.is_processing:
                        try:
                            import queue
                            result = self.streaming_service.result_queue.get(timeout=0.1)
                            
                            if not isinstance(result, tuple) or len(result) != 3:
                                print(f"⚠️ Unexpected result format: {result}")
                                continue
                            
                            transcript, is_final, is_violent = result
                            
                            if isinstance(transcript, str) and transcript.startswith("ERROR"):
                                print(f"❌ Error received: {transcript}")
                                break
                            
                            if transcript and transcript.strip():
                                violence_indicator = " ⚠️ VIOLENCE RISK DETECTED!" if is_violent else ""
                                print(f"Transcript received: '{transcript}' (final: {is_final}, risk: {is_violent}){violence_indicator}")
                                
                                transcript_clean = transcript.strip()
                                
                                if is_final:
                                    if transcript_clean and transcript_clean not in [p.strip() for p in self.transcript_parts]:
                                        self.transcript_parts.append(transcript_clean)
                                    
                                    self.last_partial_transcript = ""
                                    self.current_transcript = " ".join(self.transcript_parts).strip()
                                    print(f"Transcript updated (final): {self.current_transcript}")
                                else:
                                    self.last_partial_transcript = transcript_clean
                                    
                                    base_text = " ".join(self.transcript_parts).strip()
                                    if base_text:
                                        self.current_transcript = base_text + " " + self.last_partial_transcript
                                    else:
                                        self.current_transcript = self.last_partial_transcript
                                    
                                    print(f"Transcript updated (partial): {self.current_transcript}")
                            elif transcript:
                                print("Empty transcript received")
                        except queue.Empty:
                            if not self.streaming_service.is_streaming:
                                print("⚠️ Streaming stopped, exiting receive loop")
                                break
                            continue
                        except ValueError as e:
                            print(f"❌ Error unpacking result: {e}")
                            print(f"   Result received: {result if 'result' in locals() else 'N/A'}")
                            import traceback
                            traceback.print_exc()
                            continue
                        except Exception as e:
                            if self.is_processing:
                                print(f"❌ Error receiving transcripts: {e}")
                                import traceback
                                traceback.print_exc()
                            break
                except Exception as e:
                    if self.is_processing:
                        print(f"Error receiving transcripts: {e}")
                        import traceback
                        traceback.print_exc()
            
            send_thread = threading.Thread(target=send_audio, daemon=True)
            self.transcript_thread = threading.Thread(target=receive_transcripts, daemon=True)
            
            send_thread.start()
            self.transcript_thread.start()
            
        except Exception as e:
            self.is_processing = False
            self._cleanup_audio()
            raise RuntimeError(f"Error starting capture: {str(e)}")
    
    def _cleanup_audio(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
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
            import wave
            import os
            from datetime import datetime
            
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(temp_dir, filename)
            
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            
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
            import numpy as np
            
            audio_bytes = b''.join(self.recorded_audio_frames)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            sample_rate = 16000
            downsampled = audio_float[::10]
            times = np.arange(len(downsampled)) / (sample_rate / 10)
            
            return {
                'times': times.tolist(),
                'values': downsampled.tolist(),
                'sample_rate': sample_rate
            }
        except Exception as e:
            print(f"Error processing waveform: {e}")
            return None
    
    def get_current_transcript(self) -> str:
        return self.current_transcript or "Waiting for transcription..."
    
    @staticmethod
    def list_audio_devices():
        devices = []
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            
            for i in range(device_count):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append((i, info['name']))
            
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
