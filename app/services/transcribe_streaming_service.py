import asyncio
import os
import queue
import threading
import traceback
from typing import Optional, Generator, Tuple, Any

from config.constants import (
    CHUNK_100MS_BYTES,
    CONTEXT_WINDOW_SIZE,
    MIN_TEXT_LENGTH_VIOLENCE,
    SENDER_SLEEP_SEC,
    VIOLENCE_MAX_INPUT_CHARS,
    VIOLENCE_MAX_LENGTH,
    VIOLENCE_THRESHOLD,
)
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from transformers import pipeline
import torch


class ZeroShotViolenceDetector:
    """Two-stage violence/risk detector.

    Stage 1 — binary: asks the model a single yes/no question ("is this
    dangerous?") using a well-designed NLI hypothesis.  With only two
    competing labels the model concentrates its probability mass, giving
    much higher recall for indirect or third-person descriptions of risk.

    Stage 2 — category: if Stage 1 flags danger, a second pass over a small
    set of broad category labels identifies the type of risk for the UI alert.
    This pass never blocks the alert; it only adds the label.

    Fast-path keywords: a compact list of completely unambiguous tokens
    (weapon names, explicit murder phrases, child exploitation terms) bypasses
    both model calls and returns immediately with confidence 1.0.
    """

    # --- Binary detection (Stage 1) ---
    _BINARY_DANGER = "situação de risco, perigo ou violência"
    _BINARY_SAFE   = "situação normal e segura"
    _BINARY_HYPOTHESIS = "Esta frase descreve uma {}."
    _BINARY_THRESHOLD = 0.55

    _CATEGORY_LABELS = [
        "violência física ou ameaça de morte",
        "perseguição ou stalking",
        "assédio verbal, psicológico ou online",
        "posse ou uso de arma",
        "exploração ou abuso sexual de menor",
        "violência doméstica",
        "ameaça velada ou coação",
        "pedido de socorro ou emergência",
        "exposição não consentida de conteúdo íntimo",
    ]
    _CATEGORY_HYPOTHESIS = "Esta frase é sobre {}."

    _FAST_KEYWORDS: dict[str, str] = {
        # weapons
        "faca": "posse ou uso de arma",
        "facaço": "posse ou uso de arma",
        "canivete": "posse ou uso de arma",
        "revólver": "posse ou uso de arma",
        "pistola": "posse ou uso de arma",
        "espingarda": "posse ou uso de arma",
        "rifle": "posse ou uso de arma",
        "cutelo": "posse ou uso de arma",
        "estou armado": "posse ou uso de arma",
        "tenho uma arma": "posse ou uso de arma",
        # explicit death threats
        "vou te matar": "violência física ou ameaça de morte",
        "te mato": "violência física ou ameaça de morte",
        "vou matar você": "violência física ou ameaça de morte",
        "você não vai sair vivo": "violência física ou ameaça de morte",
        # child exploitation
        "pedofilia": "exploração ou abuso sexual de menor",
        "abuso sexual infantil": "exploração ou abuso sexual de menor",
        "nude de menor": "exploração ou abuso sexual de menor",
        "nude de criança": "exploração ou abuso sexual de menor",
        "sexo com criança": "exploração ou abuso sexual de menor",
        "sexo com menor": "exploração ou abuso sexual de menor",
    }

    def __init__(self, use_cuda: bool = False):
        print("🚀 Initializing violence detector...")
        if use_cuda and torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            device = 0
            dtype = torch.float16
        else:
            print("⚠️ Using CPU for violence detector (CUDA disabled).")
            device = -1
            dtype = torch.float32

        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=device,
            torch_dtype=dtype,
        )

    def _fast_path(self, text: str) -> tuple[bool, str]:
        """Return (is_danger, category) for obvious keyword matches, else (False, '')."""
        text_lower = text.lower()
        for token, category in self._FAST_KEYWORDS.items():
            if token in text_lower:
                return True, category
        return False, ""

    def _classify_binary(self, text: str) -> float:
        """Return the danger probability (0-1) from the binary stage."""
        result = self.classifier(
            text,
            [self._BINARY_DANGER, self._BINARY_SAFE],
            hypothesis_template=self._BINARY_HYPOTHESIS,
            multi_label=False,
            truncation=True,
            max_length=VIOLENCE_MAX_LENGTH,
        )
        danger_idx = result["labels"].index(self._BINARY_DANGER)
        return result["scores"][danger_idx]

    def _classify_category(self, text: str) -> str:
        """Return the most likely risk category label."""
        result = self.classifier(
            text,
            self._CATEGORY_LABELS,
            hypothesis_template=self._CATEGORY_HYPOTHESIS,
            multi_label=False,
            truncation=True,
            max_length=VIOLENCE_MAX_LENGTH,
        )
        return result["labels"][0]

    def predict(self, text: str) -> Tuple[bool, str, float]:
        if len(text) < MIN_TEXT_LENGTH_VIOLENCE:
            return False, "too_short", 0.0

        is_fast, fast_category = self._fast_path(text)
        if is_fast:
            return True, fast_category, 1.0

        try:
            truncated = text[:VIOLENCE_MAX_INPUT_CHARS]
            danger_score = self._classify_binary(truncated)
            if danger_score >= self._BINARY_THRESHOLD:
                category = self._classify_category(truncated)
                return True, category, danger_score
            return False, "safe", danger_score
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            return False, "error", 0.0


class ViolenceHandler(TranscriptResultStreamHandler):
    def __init__(self, stream, output_queue):
        super().__init__(stream)
        self.output_queue = output_queue
        self.full_transcript_context = ""
        self.last_analyzed_text = ""
        self._detector_initialized = False
        self._detector = None
        self.context_window = []
        self._transcript_log_count = 0
        self._events_received = 0

    async def handle_events(self):
        async for event in self._transcript_result_stream:
            self._events_received += 1
            if self._events_received <= 2 or self._events_received % 50 == 0:
                print(f"📥 Stream event #{self._events_received}: {type(event).__name__}", flush=True)
            if isinstance(event, TranscriptEvent):
                await self.handle_transcript_event(event)

    def _initialize_detector(self):
        if not self._detector_initialized:
            try:
                self._detector = ZeroShotViolenceDetector()
                self._detector_initialized = True
            except Exception as e:
                print(f"Warning initializing detector: {e}")

    def _analyze_violence_risk(self, transcript_text: str) -> bool:
        try:
            self._initialize_detector()
            
            if self._detector is None:
                return False
            
            is_danger, _, _ = self._detector.predict(transcript_text)
            return is_danger
            
        except Exception as e:
            print(f"Error analyzing risk with detector: {e}")
            return False

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results or []
        for result in results:
            alts = result.alternatives if result.alternatives else []
            transcript = (alts[0].transcript or "").strip() if alts else ""
            is_final = not result.is_partial

            try:
                self.output_queue.put_nowait((transcript, is_final, False))
                self._transcript_log_count += 1
                if transcript and (self._transcript_log_count <= 3 or self._transcript_log_count % 20 == 0):
                    txt = (transcript[:50] + "…") if len(transcript) > 50 else transcript
                    print(f"📝 AWS returned: {'[final]' if is_final else '[partial]'} \"{txt}\"", flush=True)
            except queue.Full:
                pass

            if is_final and transcript:
                self.context_window.append(transcript.strip())
                if len(self.context_window) > CONTEXT_WINDOW_SIZE:
                    self.context_window.pop(0)

            if not is_final:
                continue
            current_context = " ".join(self.context_window)
            text_to_analyze = f"{current_context} {transcript}".strip()
            if len(text_to_analyze) > MIN_TEXT_LENGTH_VIOLENCE and text_to_analyze != self.last_analyzed_text:
                self.last_analyzed_text = text_to_analyze

                async def check_violence():
                    if self._detector is None:
                        return
                    try:
                        is_violent, label, score = await asyncio.to_thread(
                            self._detector.predict, text_to_analyze
                        )
                        if is_violent:
                            print(f"🚨 VIOLENCE ALERT: {label} ({score:.2f}) -> {text_to_analyze}", flush=True)
                            try:
                                self.output_queue.put_nowait(
                                    (f"__VIOLENCE_ALERT__:{label}|{score:.2f}", False, True)
                                )
                            except queue.Full:
                                pass
                    except Exception as e:
                        print(f"Violence detection error: {e}", flush=True)

                asyncio.create_task(check_violence())


class TranscribeStreamingService:
    DEFAULT_LANGUAGE_CODE = 'pt-BR'
    SAMPLE_RATE = 16000

    # Shared detector instance across all sessions
    _shared_detector: Optional[ZeroShotViolenceDetector] = None
    _detector_lock = threading.Lock()

    @classmethod
    def _get_or_create_detector(cls) -> Optional[ZeroShotViolenceDetector]:
        if cls._shared_detector is not None:
            return cls._shared_detector
        with cls._detector_lock:
            if cls._shared_detector is None:
                try:
                    cls._shared_detector = ZeroShotViolenceDetector()
                except Exception as e:
                    print(f"Warning: could not load violence detector: {e}")
                    return None
        return cls._shared_detector

    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.result_queue = queue.Queue()
        self.input_queue_sync = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = None
        self._is_streaming_active = False
        threading.Thread(
            target=self._get_or_create_detector,
            daemon=True,
            name="detector-warmup",
        ).start()

    @property
    def is_streaming(self):
        return self._is_streaming_active and not self.stop_event.is_set()

    def start_stream(self, language_code: str = DEFAULT_LANGUAGE_CODE):
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.stop_event.clear()
        self._is_streaming_active = True
        
        with self.input_queue_sync.mutex:
            self.input_queue_sync.queue.clear()
        with self.result_queue.mutex:
            self.result_queue.queue.clear()
        
        self.processing_thread = threading.Thread(
            target=self._run_async_loop,
            args=(language_code,),
            daemon=True
        )
        self.processing_thread.start()

    def _run_async_loop(self, language_code):
        try:
            asyncio.run(self._worker(language_code))
        except Exception as e:
            print(f"Fatal error in AWS streaming thread: {e}", flush=True)
            traceback.print_exc()
            self.result_queue.put((f"ERROR: {str(e)}", False, False))
        finally:
            self._is_streaming_active = False

    async def _worker(self, language_code):
        print("⏳ Connecting to AWS Transcribe...", flush=True)
        client = TranscribeStreamingClient(region=self.region_name)
        handler = ViolenceHandler(None, self.result_queue)
        handler._detector = self._shared_detector
        handler._detector_initialized = self._shared_detector is not None
        try:
            stream = await client.start_stream_transcription(
                language_code=language_code,
                media_sample_rate_hz=self.SAMPLE_RATE,
                media_encoding="pcm"
            )
            handler._transcript_result_stream = stream.output_stream
            print("✅ AWS stream active. Starting transcript reception.", flush=True)
            events_sent = [0]

            async def sender():
                try:
                    buffer = b""
                    while not self.stop_event.is_set():
                        try:
                            chunk = await asyncio.to_thread(
                                self.input_queue_sync.get, timeout=0.05
                            )
                            if chunk is None:
                                break
                            buffer += chunk
                            while len(buffer) >= CHUNK_100MS_BYTES:
                                to_send = buffer[:CHUNK_100MS_BYTES]
                                buffer = buffer[CHUNK_100MS_BYTES:]
                                await stream.input_stream.send_audio_event(audio_chunk=to_send)
                                events_sent[0] += 1
                                if events_sent[0] == 1:
                                    print("📤 First audio chunk sent to AWS.", flush=True)
                                elif events_sent[0] % 100 == 0:
                                    print(f"📤 Sent {events_sent[0]} audio events to AWS.", flush=True)
                                await asyncio.sleep(SENDER_SLEEP_SEC)
                        except queue.Empty:
                            continue
                    if buffer:
                        await stream.input_stream.send_audio_event(audio_chunk=buffer)
                except Exception as e:
                    print(f"Sender error: {e}", flush=True)
                    traceback.print_exc()
                finally:
                    await stream.input_stream.end_stream()
                    print(f"📤 Total audio events sent: {events_sent[0]}", flush=True)

            await asyncio.gather(sender(), handler.handle_events())

        except Exception as e:
            error_msg = f"ERROR: Connection or processing failure: {str(e)}"
            print(error_msg, flush=True)
            traceback.print_exc()
            self.result_queue.put((error_msg, False, False))
        finally:
            self._is_streaming_active = False
            print("🏁 Stream finished.", flush=True)

    def send_audio_chunk(self, audio_chunk: bytes):
        if self.is_streaming:
            self.input_queue_sync.put(audio_chunk)

    def stop_stream(self):
        self.stop_event.set()
        self.input_queue_sync.put(None)
        self._is_streaming_active = False

    def transcribe_audio_stream(
        self,
        audio_stream: Generator[bytes, None, None],
        language_code: str = DEFAULT_LANGUAGE_CODE
    ) -> Generator[Tuple[str, bool, bool], None, None]:
        self.start_stream(language_code)
        
        def consume_input():
            try:
                for chunk in audio_stream:
                    if not self.is_streaming: break
                    self.send_audio_chunk(chunk)
            except Exception as e:
                print(f"Error consuming audio: {e}")
            finally:
                self.stop_stream()

        input_thread = threading.Thread(target=consume_input, daemon=True)
        input_thread.start()
        
        while self.is_streaming or not self.result_queue.empty():
            try:
                result = self.result_queue.get(timeout=0.1)
                
                if isinstance(result[0], str) and result[0].startswith("ERROR"):
                    print(result[0])
                    yield result[0], True, False
                    break
                    
                yield result
                
            except queue.Empty:
                if not self.processing_thread.is_alive() and self.result_queue.empty():
                    break
                continue
