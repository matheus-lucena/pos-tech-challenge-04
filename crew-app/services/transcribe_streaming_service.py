"""Serviço para transcrição de áudio em tempo real usando AWS Transcribe SDK e detecção de violência com LLM."""

import asyncio
import queue
import threading
import os
from typing import Optional, Generator, Tuple, Any
from dotenv import load_dotenv

# SDK Oficial da AWS
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# LLM para análise de risco
import litellm

load_dotenv()

# Configuração do LLM
LLM_MODEL = "ollama/qwen2.5:7b"
LLM_BASE_URL = "http://localhost:11434"

class ViolenceHandler(TranscriptResultStreamHandler):
    """Callback interno que processa o texto vindo da AWS e analisa risco com LLM."""
    
    def __init__(self, stream, output_queue):
        super().__init__(stream)
        self.output_queue = output_queue
        self.full_transcript_context = ""  # Mantém contexto completo para análise
        self._llm_initialized = False
        self.last_analyzed_text = ""  # Evita analisar o mesmo texto múltiplas vezes
        self.last_analyzed_text = ""  # Evita analisar o mesmo texto múltiplas vezes

    def _initialize_llm(self):
        """Inicializa o LLM se ainda não foi inicializado."""
        if not self._llm_initialized:
            try:
                os.environ["OPENAI_API_KEY"] = "sk-ollama-local"
                self._llm_initialized = True
            except Exception as e:
                print(f"⚠️ Aviso ao inicializar LLM: {e}")

    def _analyze_violence_risk(self, transcript_text: str) -> bool:
        """
        Analisa o risco de violência usando Ollama LLM.
        
        Args:
            transcript_text: Texto transcrito para analisar
            
        Returns:
            True se houver risco de violência, False caso contrário
        """
        try:
            self._initialize_llm()
            
            # Prompt para análise de risco
            prompt = f"""Analise o seguinte texto e determine se há indícios de violência, perigo iminente, ou necessidade de ajuda urgente.

Texto: "{transcript_text}"

Responda APENAS com "SIM" se houver risco de violência ou perigo, ou "NÃO" se não houver risco.
Resposta:"""
            
            # Chama o LLM via LiteLLM
            response = litellm.completion(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                api_base=LLM_BASE_URL,
                timeout=5.0  # Timeout curto para não bloquear
            )
            
            # Extrai a resposta
            answer = response.choices[0].message.content.strip().upper()
            
            # Verifica se a resposta indica violência
            is_violent = "SIM" in answer or "YES" in answer or "RISCO" in answer or "PERIGO" in answer
            
            return is_violent
            
        except Exception as e:
            # Em caso de erro, retorna False para não bloquear a transcrição
            print(f"⚠️ Erro ao analisar risco com LLM: {e}")
            return False

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if not result.alternatives:
                continue
            
            transcript = result.alternatives[0].transcript
            is_final = not result.is_partial
            
            # Atualiza contexto completo para análise mais precisa
            if transcript.strip():
                if is_final:
                    # Transcrição final - adiciona ao contexto
                    self.full_transcript_context += " " + transcript.strip()
                    text_to_analyze = self.full_transcript_context
                else:
                    # Transcrição parcial - usa contexto temporário
                    text_to_analyze = (self.full_transcript_context + " " + transcript.strip()).strip()
            else:
                # Transcrição vazia - usa contexto atual
                text_to_analyze = self.full_transcript_context
            
            # Analisa risco de violência usando LLM
            
            # Só analisa se houver texto significativo (mais de 3 caracteres) e for diferente do último analisado
            is_violent = False
            if len(text_to_analyze.strip()) > 3 and text_to_analyze.strip() != self.last_analyzed_text:
                # Analisa em thread separada para não bloquear
                try:
                    # Executa análise LLM de forma assíncrona
                    is_violent = await asyncio.to_thread(self._analyze_violence_risk, text_to_analyze)
                    self.last_analyzed_text = text_to_analyze.strip()
                except Exception as e:
                    print(f"⚠️ Erro ao executar análise LLM: {e}")
            elif text_to_analyze.strip() == self.last_analyzed_text:
                # Reutiliza resultado anterior se for o mesmo texto
                # (útil para transcrições parciais que se repetem)
                pass
            
            # Envia para a fila: (texto, é_final, é_violencia)
            # SEMPRE envia, mesmo que seja parcial, para garantir que tudo seja mostrado
            try:
                self.output_queue.put_nowait((transcript.strip(), is_final, is_violent))
            except queue.Full:
                # Se a fila estiver cheia, tenta novamente (não deve acontecer, mas por segurança)
                pass

class TranscribeStreamingService:
    """
    Serviço adaptado para transcrição e detecção de violência.
    """
    
    DEFAULT_LANGUAGE_CODE = 'pt-BR'
    SAMPLE_RATE = 16000
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.result_queue = queue.Queue()  # Fila de saída de texto
        self.input_queue_sync = queue.Queue() # Fila de entrada de áudio
        self.stop_event = threading.Event()
        self.processing_thread = None
        self._is_streaming_active = False # Flag interna para compatibilidade

    @property
    def is_streaming(self):
        """Propriedade para manter compatibilidade com seu código de UI."""
        return self._is_streaming_active and not self.stop_event.is_set()

    def start_stream(self, language_code: str = DEFAULT_LANGUAGE_CODE):
        """Inicia a thread de processamento em background."""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.stop_event.clear()
        self._is_streaming_active = True
        
        # Limpa filas antigas
        with self.input_queue_sync.mutex:
            self.input_queue_sync.queue.clear()
        with self.result_queue.mutex:
            self.result_queue.queue.clear()
        
        # Inicia o loop async em uma thread separada
        self.processing_thread = threading.Thread(
            target=self._run_async_loop,
            args=(language_code,),
            daemon=True
        )
        self.processing_thread.start()

    def _run_async_loop(self, language_code):
        """Wrapper para rodar o asyncio dentro da Thread."""
        try:
            asyncio.run(self._worker(language_code))
        except Exception as e:
            print(f"Erro fatal na thread do AWS: {e}")
            self.result_queue.put((f"ERROR: {str(e)}", False, False))
        finally:
            self._is_streaming_active = False

    async def _worker(self, language_code):
        """Lógica pesada do Asyncio + AWS SDK."""
        client = TranscribeStreamingClient(region=self.region_name)

        try:
            stream = await client.start_stream_transcription(
                language_code=language_code,
                media_sample_rate_hz=self.SAMPLE_RATE,
                media_encoding="pcm"
            )

            # CORREÇÃO 2: Passamos stream.output_stream E a fila
            handler = ViolenceHandler(stream.output_stream, self.result_queue)

            async def sender():
                """Lê da fila síncrona e envia para AWS async."""
                while not self.stop_event.is_set():
                    try:
                        # Non-blocking get da fila thread-safe
                        try:
                            chunk = self.input_queue_sync.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.01) # Evita CPU 100%
                            continue

                        if chunk is None: 
                            break
                        
                        await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    except Exception as e:
                        print(f"Erro no sender: {e}")
                        break
                await stream.input_stream.end_stream()

            # Roda envio e recebimento em paralelo
            await asyncio.gather(sender(), handler.handle_events())

        except Exception as e:
            self.result_queue.put((f"ERROR: Falha na conexão AWS: {str(e)}", False, False))

    def send_audio_chunk(self, audio_chunk: bytes):
        """Coloca o áudio na fila de processamento (Thread-Safe)."""
        if self.is_streaming:
            self.input_queue_sync.put(audio_chunk)

    def stop_stream(self):
        """Para o serviço."""
        self.stop_event.set()
        self.input_queue_sync.put(None) # Sinal de parada para o sender
        self._is_streaming_active = False

    def transcribe_audio_stream(
        self,
        audio_stream: Generator[bytes, None, None],
        language_code: str = DEFAULT_LANGUAGE_CODE
    ) -> Generator[Tuple[str, bool, bool], None, None]:
        """
        Método principal consumido pelo UI.
        Retorna generator de: (texto, is_final, is_violence_detected)
        """
        self.start_stream(language_code)
        
        # Thread auxiliar para consumir o generator de entrada (ex: microfone)
        def consume_input():
            try:
                for chunk in audio_stream:
                    if not self.is_streaming: break
                    self.send_audio_chunk(chunk)
            except Exception as e:
                print(f"Erro ao consumir áudio: {e}")
            finally:
                self.stop_stream()

        input_thread = threading.Thread(target=consume_input, daemon=True)
        input_thread.start()
        
        # Generator de Saída para a UI
        while self.is_streaming or not self.result_queue.empty():
            try:
                # Timeout curto para permitir checar o flag is_streaming
                result = self.result_queue.get(timeout=0.1)
                
                # Se for mensagem de erro
                if isinstance(result[0], str) and result[0].startswith("ERROR"):
                    print(result[0])
                    yield result[0], True, False
                    break
                    
                yield result
                
            except queue.Empty:
                if not self.processing_thread.is_alive() and self.result_queue.empty():
                    break
                continue