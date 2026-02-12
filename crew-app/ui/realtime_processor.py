"""Processador para transcrição de áudio em tempo real."""

import threading
import time
import pyaudio
import numpy as np
from typing import Optional
from services.transcribe_streaming_service import TranscribeStreamingService
from services.s3_service import S3Service


class RealtimeAudioProcessor:
    """Processador para áudio em tempo real."""
    
    def __init__(self):
        """Inicializa o processador."""
        self.streaming_service = TranscribeStreamingService()
        self.s3_service = S3Service()
        self.is_processing = False
        self.current_transcript = ""
        self.transcript_parts = []
        self.audio_stream = None
        self.pyaudio_instance = None
        self.transcript_thread = None
        self.recorded_audio_frames = []  # Armazena frames de áudio gravados
        self.audio_file_path = None  # Caminho do arquivo de áudio salvo
    
    def start_realtime_transcription(
        self,
        audio_file_path: Optional[str] = None,
        language_code: str = "pt-BR"
    ) -> str:
        """
        Inicia transcrição em tempo real.
        
        Args:
            audio_file_path: Caminho do arquivo de áudio (se fornecido)
            language_code: Código do idioma
        
        Returns:
            Mensagem de status
        """
        if self.is_processing:
            return "⚠️ Já existe uma transcrição em andamento. Pare a anterior primeiro."
        
        self.is_processing = True
        self.current_transcript = ""
        self.transcript_parts = []
        
        try:
            if audio_file_path:
                # Processa arquivo de áudio
                return self._process_audio_file(audio_file_path, language_code)
            else:
                # Inicia stream do microfone
                return self._start_microphone_stream(language_code)
        except Exception as e:
            self.is_processing = False
            return f"❌ Erro ao iniciar transcrição: {str(e)}"
    
    def _process_audio_file(
        self,
        audio_file_path: str,
        language_code: str
    ) -> str:
        """Processa um arquivo de áudio em tempo real."""
        import librosa
        import numpy as np
        
        try:
            # Carrega áudio
            y, sr = librosa.load(audio_file_path, sr=16000, mono=True)
            
            # Converte para PCM 16-bit
            audio_int16 = (y * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Divide em chunks
            chunk_size = 1024 * 2  # 2 bytes por sample
            chunks = [
                audio_bytes[i:i+chunk_size]
                for i in range(0, len(audio_bytes), chunk_size)
            ]
            
            # Inicia stream
            transcript_gen = self.streaming_service.start_stream(language_code)
            
            # Thread para enviar chunks
            def send_chunks():
                for chunk in chunks:
                    if not self.is_processing:
                        break
                    self.streaming_service.send_audio_chunk(chunk)
                    time.sleep(0.01)  # Pequeno delay entre chunks
                self.streaming_service.stop_stream()
            
            send_thread = threading.Thread(target=send_chunks, daemon=True)
            send_thread.start()
            
            # Coleta transcrições
            full_transcript = ""
            for transcript, is_final in transcript_gen:
                if not self.is_processing:
                    break
                if transcript:
                    if is_final:
                        full_transcript += transcript + " "
                        self.transcript_parts.append(transcript)
                    else:
                        # Transcrição parcial
                        current_partial = transcript
            
            self.current_transcript = full_transcript.strip()
            self.is_processing = False
            
            return f"✅ Transcrição concluída: {len(self.transcript_parts)} segmentos processados."
            
        except Exception as e:
            self.is_processing = False
            return f"❌ Erro ao processar arquivo: {str(e)}"
    
    def start_microphone_streaming(self, language_code: str = "pt-BR", device_index: Optional[int] = None):
        """
        Inicia captura e transcrição do microfone em tempo real.
        
        Args:
            language_code: Código do idioma
            device_index: Índice do dispositivo de áudio (None para padrão)
        """
        if self.is_processing:
            raise RuntimeError("Já existe uma transcrição em andamento.")
        
        self.is_processing = True
        self.current_transcript = ""
        self.transcript_parts = []  # Lista de partes finais confirmadas
        self.last_partial_transcript = ""  # Última transcrição parcial recebida
        self.recorded_audio_frames = []  # Limpa frames anteriores
        
        # Configurações de áudio
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        try:
            # Inicializa PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Verifica o dispositivo se especificado
            if device_index is not None:
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
                    print(f"🎤 Usando dispositivo: {device_info['name']}")
                    print(f"   Taxa de amostragem suportada: {device_info.get('defaultSampleRate', 'N/A')}")
                except Exception as e:
                    print(f"⚠️ Aviso ao verificar dispositivo: {e}")
            
            # Abre stream de áudio
            self.audio_stream = self.pyaudio_instance.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_index  # Usa o dispositivo especificado
            )
            
            print(f"✅ Stream de áudio aberto: {CHANNELS} canal(is), {RATE}Hz, {FORMAT}, {CHUNK} frames por buffer")
            
            # Inicia stream de transcrição
            self.streaming_service.start_stream(language_code)
            
            # Aguarda um pouco antes de começar a enviar áudio
            time.sleep(0.5)
            
            # Thread para enviar áudio
            def send_audio():
                try:
                    chunks_sent = 0
                    print(f"🎤 Iniciando envio de áudio (CHUNK: {CHUNK} bytes)")
                    
                    while self.is_processing:
                        try:
                            # Verifica se o streaming ainda está ativo
                            if not self.streaming_service.is_streaming:
                                print("⚠️ Streaming parado, interrompendo envio")
                                break
                            
                            # Lê chunk do microfone
                            audio_data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
                            
                            if len(audio_data) == 0:
                                print("⚠️ Chunk de áudio vazio recebido")
                                continue
                            
                            # Verifica se o tamanho está correto (deve ser múltiplo de 2 para 16-bit)
                            if len(audio_data) % 2 != 0:
                                print(f"⚠️ Tamanho de chunk inválido: {len(audio_data)} (deve ser par)")
                                continue
                            
                            # Salva o frame para reprodução posterior
                            self.recorded_audio_frames.append(audio_data)
                            
                            # Verifica se ainda está processando antes de enviar
                            if self.is_processing and self.streaming_service.is_streaming:
                                # Converte para bytes e envia
                                self.streaming_service.send_audio_chunk(audio_data)
                                chunks_sent += 1
                                if chunks_sent % 20 == 0:  # Log a cada 20 chunks
                                    print(f"📤 Enviados {chunks_sent} chunks de áudio ({len(audio_data)} bytes cada)")
                            else:
                                print("⚠️ Streaming não está ativo, parando envio")
                                break
                            
                            # Não precisa de sleep aqui, o read já bloqueia
                        except Exception as e:
                            if self.is_processing:
                                print(f"❌ Erro ao enviar áudio: {e}")
                                import traceback
                                traceback.print_exc()
                            break
                    print(f"✅ Envio de áudio finalizado. Total de chunks: {chunks_sent}")
                finally:
                    # Para o stream apenas se ainda estiver ativo
                    if self.streaming_service.is_streaming:
                        self.streaming_service.stop_stream()
            
            # Thread para receber transcrições
            def receive_transcripts():
                try:
                    print("Iniciando recebimento de transcrições...")
                    while self.is_processing:
                        try:
                            # O novo serviço usa uma fila, então precisamos consumir dela
                            # Usa timeout curto para não bloquear indefinidamente
                            import queue
                            result = self.streaming_service.result_queue.get(timeout=0.1)
                            
                            # Verifica o formato do resultado
                            if not isinstance(result, tuple) or len(result) != 3:
                                print(f"⚠️ Formato de resultado inesperado: {result}")
                                continue
                            
                            transcript, is_final, is_violent = result
                            
                            # Verifica se é erro
                            if isinstance(transcript, str) and transcript.startswith("ERROR"):
                                print(f"❌ Erro recebido: {transcript}")
                                break
                            
                            if transcript and transcript.strip():
                                violence_indicator = " ⚠️ RISCO DE VIOLÊNCIA DETECTADO!" if is_violent else ""
                                print(f"Transcrição recebida: '{transcript}' (final: {is_final}, risco: {is_violent}){violence_indicator}")
                                
                                transcript_clean = transcript.strip()
                                
                                if is_final:
                                    # Transcrição final - adiciona ao texto completo
                                    # Verifica se já não foi adicionada (evita duplicatas)
                                    if transcript_clean and transcript_clean not in [p.strip() for p in self.transcript_parts]:
                                        self.transcript_parts.append(transcript_clean)
                                    
                                    # Limpa a transcrição parcial já que foi finalizada
                                    self.last_partial_transcript = ""
                                    
                                    # Reconstrói o texto completo apenas com partes finais
                                    self.current_transcript = " ".join(self.transcript_parts).strip()
                                    print(f"Transcrição atualizada (final): {self.current_transcript}")
                                else:
                                    # Transcrição parcial - atualiza a última parcial
                                    # O AWS envia a transcrição parcial completa, não apenas o incremento
                                    self.last_partial_transcript = transcript_clean
                                    
                                    # Constrói texto: partes finais + última parcial
                                    base_text = " ".join(self.transcript_parts).strip()
                                    if base_text:
                                        self.current_transcript = base_text + " " + self.last_partial_transcript
                                    else:
                                        self.current_transcript = self.last_partial_transcript
                                    
                                    print(f"Transcrição atualizada (parcial): {self.current_transcript}")
                            elif transcript:
                                # Transcrição vazia ou só espaços
                                print("Transcrição vazia recebida")
                        except queue.Empty:
                            # Timeout - continua verificando
                            if not self.streaming_service.is_streaming:
                                print("⚠️ Streaming parado, saindo do loop de recebimento")
                                break
                            continue
                        except ValueError as e:
                            # Erro de desempacotamento
                            print(f"❌ Erro ao desempacotar resultado: {e}")
                            print(f"   Resultado recebido: {result if 'result' in locals() else 'N/A'}")
                            import traceback
                            traceback.print_exc()
                            continue
                        except Exception as e:
                            if self.is_processing:
                                print(f"❌ Erro ao receber transcrições: {e}")
                                import traceback
                                traceback.print_exc()
                            break
                except Exception as e:
                    if self.is_processing:
                        print(f"Erro ao receber transcrições: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Inicia threads
            send_thread = threading.Thread(target=send_audio, daemon=True)
            self.transcript_thread = threading.Thread(target=receive_transcripts, daemon=True)
            
            send_thread.start()
            self.transcript_thread.start()
            
        except Exception as e:
            self.is_processing = False
            self._cleanup_audio()
            raise RuntimeError(f"Erro ao iniciar captura: {str(e)}")
    
    def _cleanup_audio(self):
        """Limpa recursos de áudio."""
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
        """Para a transcrição em tempo real."""
        if not self.is_processing:
            return "⚠️ Nenhuma transcrição em andamento."
        
        self.is_processing = False
        self.streaming_service.stop_stream()
        
        # Salva o áudio gravado
        self._save_recorded_audio()
        
        self._cleanup_audio()
        
        # Aguarda threads terminarem
        if self.transcript_thread:
            self.transcript_thread.join(timeout=2)
        
        return "⏹️ Transcrição parada."
    
    def _save_recorded_audio(self):
        """Salva o áudio gravado em um arquivo WAV."""
        if not self.recorded_audio_frames:
            return None
        
        try:
            import wave
            import os
            from datetime import datetime
            
            # Cria diretório temporário se não existir
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Gera nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(temp_dir, filename)
            
            # Salva como WAV
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(16000)  # 16kHz
            
            # Escreve todos os frames
            for frame in self.recorded_audio_frames:
                wf.writeframes(frame)
            
            wf.close()
            self.audio_file_path = filepath
            return filepath
        except Exception as e:
            print(f"Erro ao salvar áudio: {e}")
            return None
    
    def get_recorded_audio_path(self) -> Optional[str]:
        """Retorna o caminho do áudio gravado."""
        return self.audio_file_path
    
    def get_audio_waveform_data(self):
        """Retorna dados da forma de onda para visualização."""
        if not self.recorded_audio_frames:
            return None
        
        try:
            import numpy as np
            
            # Concatena todos os frames
            audio_bytes = b''.join(self.recorded_audio_frames)
            
            # Converte para numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Normaliza para float32 entre -1 e 1
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Retorna dados para visualização (amostras e valores)
            # Para visualização, podemos retornar uma amostra reduzida
            sample_rate = 16000
            # Pega uma amostra a cada 10 para reduzir o tamanho
            downsampled = audio_float[::10]
            times = np.arange(len(downsampled)) / (sample_rate / 10)
            
            return {
                'times': times.tolist(),
                'values': downsampled.tolist(),
                'sample_rate': sample_rate
            }
        except Exception as e:
            print(f"Erro ao processar forma de onda: {e}")
            return None
    
    def get_current_transcript(self) -> str:
        """Retorna a transcrição atual."""
        return self.current_transcript or "Aguardando transcrição..."
    
    @staticmethod
    def list_audio_devices():
        """
        Lista todos os dispositivos de áudio disponíveis.
        
        Returns:
            Lista de tuplas (índice, nome) dos dispositivos de entrada
        """
        devices = []
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            
            for i in range(device_count):
                info = p.get_device_info_by_index(i)
                # Apenas dispositivos de entrada
                if info['maxInputChannels'] > 0:
                    devices.append((i, info['name']))
            
            p.terminate()
        except Exception as e:
            print(f"Erro ao listar dispositivos: {e}")
        
        return devices


# Instância global
_realtime_processor = RealtimeAudioProcessor()


def processar_audio_realtime(audio_file: Optional[str]) -> tuple:
    """
    Processa áudio em tempo real.
    
    Args:
        audio_file: Caminho do arquivo de áudio
    
    Returns:
        Tupla (transcrição, status)
    """
    if not audio_file:
        return "Nenhum áudio fornecido.", "⚠️"
    
    try:
        status = _realtime_processor.start_realtime_transcription(audio_file)
        transcript = _realtime_processor.get_current_transcript()
        return transcript, status
    except Exception as e:
        return f"Erro: {str(e)}", "❌"



