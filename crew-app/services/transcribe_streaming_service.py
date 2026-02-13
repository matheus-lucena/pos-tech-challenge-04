import asyncio
import queue
import threading
import os
from typing import Optional, Generator, Tuple, Any
from dotenv import load_dotenv
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from transformers import pipeline
import torch

load_dotenv()


class ZeroShotViolenceDetector:
    def __init__(self):
        print("🚀 Inicializando Detector de Violência...")
        
        if not torch.cuda.is_available():
            print("⚠️ ALERTA: CUDA não detectado! O script vai rodar na CPU. Verifique sua instalação do PyTorch.")
            device = -1
        else:
            print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
            device = 0
        
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=device,
            torch_dtype=torch.float16 if device >= 0 and torch.cuda.is_available() else torch.float32
        )
        self.labels = [
            "agressão física violência", 
            "ameaça de morte perigo", 
            "menção a arma faca objeto perigoso",
            "ameaça com arma ou faca",
            "conversa tranquila normal", 
            "pedido de socorro emergência",
            "discussão verbal acalorada",
            "contexto de jogo ou filme"
        ]
        
        self.danger_keywords = [
            "faca", "facas", "facaço", "canivete",
            "arma", "armas", "revólver", "pistola", "espingarda", "rifle",
            "tiro", "atirar", "disparar", "disparo",
            "bala", "balas", "munição",
            "cutelo", "machado", "tesoura grande",
            "golpear com", "esfaquear", "atacar com",
            "tenho uma", "tenho um", "estou com", "estou armado"
        ]
        
        self.threshold = 0.75 

    def _check_danger_keywords(self, text: str) -> bool:
        """
        Verifica se o texto contém palavras-chave perigosas relacionadas a armas/facas
        """
        text_lower = text.lower()
        for keyword in self.danger_keywords:
            if keyword in text_lower:
                return True
        return False

    def predict(self, text: str) -> Tuple[bool, str, float]:
        """
        Inferência ultra-rápida com verificação híbrida (modelo + palavras-chave)
        Retorna: (is_danger, top_label, score)
        """
        if len(text) < 10: 
            return False, "muito curto", 0.0

        has_danger_keywords = self._check_danger_keywords(text)
        
        try:
            result = self.classifier(text, self.labels, multi_label=False)
            
            top_label = result['labels'][0]
            score = result['scores'][0]
            
            danger_labels = [
                "agressão física violência", 
                "ameaça de morte perigo", 
                "menção a arma faca objeto perigoso",
                "ameaça com arma ou faca",
                "pedido de socorro emergência"
            ]
            
            is_danger = (
                has_danger_keywords or 
                (top_label in danger_labels and score > self.threshold)
            )
            
            if has_danger_keywords and not is_danger:
                is_danger = True
                top_label = "menção a arma faca objeto perigoso"
                score = 0.9
            
            return is_danger, top_label, score
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            if has_danger_keywords:
                return True, "menção a arma faca objeto perigoso", 0.9
            return False, "erro", 0.0


class ViolenceHandler(TranscriptResultStreamHandler):
    def __init__(self, stream, output_queue):
        super().__init__(stream)
        self.output_queue = output_queue
        self.full_transcript_context = ""
        self.last_analyzed_text = ""
        self._detector_initialized = False
        self._detector = None

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
        results = transcript_event.transcript.results
        for result in results:
            if not result.alternatives:
                continue
            
            transcript = result.alternatives[0].transcript
            is_final = not result.is_partial
            
            if transcript.strip():
                if is_final:
                    self.full_transcript_context += " " + transcript.strip()
                    text_to_analyze = self.full_transcript_context
                else:
                    text_to_analyze = (self.full_transcript_context + " " + transcript.strip()).strip()
            else:
                text_to_analyze = self.full_transcript_context
            
            is_violent = False
            if len(text_to_analyze.strip()) > 3 and text_to_analyze.strip() != self.last_analyzed_text:
                try:
                    is_violent = await asyncio.to_thread(self._analyze_violence_risk, text_to_analyze)
                    self.last_analyzed_text = text_to_analyze.strip()
                except Exception as e:
                    print(f"Error executing violence detection: {e}")
            
            try:
                self.output_queue.put_nowait((transcript.strip(), is_final, is_violent))
            except queue.Full:
                pass


class TranscribeStreamingService:
    DEFAULT_LANGUAGE_CODE = 'pt-BR'
    SAMPLE_RATE = 16000
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.result_queue = queue.Queue()
        self.input_queue_sync = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = None
        self._is_streaming_active = False

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
            print(f"Fatal error in AWS thread: {e}")
            self.result_queue.put((f"ERROR: {str(e)}", False, False))
        finally:
            self._is_streaming_active = False

    async def _worker(self, language_code):
        client = TranscribeStreamingClient(region=self.region_name)

        try:
            stream = await client.start_stream_transcription(
                language_code=language_code,
                media_sample_rate_hz=self.SAMPLE_RATE,
                media_encoding="pcm"
            )

            handler = ViolenceHandler(stream.output_stream, self.result_queue)

            async def sender():
                while not self.stop_event.is_set():
                    try:
                        try:
                            chunk = self.input_queue_sync.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue

                        if chunk is None: 
                            break
                        
                        await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    except Exception as e:
                        print(f"Error in sender: {e}")
                        break
                await stream.input_stream.end_stream()

            await asyncio.gather(sender(), handler.handle_events())

        except Exception as e:
            self.result_queue.put((f"ERROR: AWS connection failure: {str(e)}", False, False))

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
