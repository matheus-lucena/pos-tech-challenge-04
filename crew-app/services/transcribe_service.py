"""Serviço para interação com AWS Transcribe."""

import json
import time
import urllib.request
from typing import Optional, Tuple
import boto3


class TranscribeService:
    """Serviço para transcrição de áudio usando AWS Transcribe."""
    
    # Formatos de mídia suportados
    SUPPORTED_FORMATS = ['mp3', 'mp4', 'wav', 'flac', 'ogg', 'amr', 'webm']
    
    # Configurações padrão
    DEFAULT_LANGUAGE_CODE = 'pt-BR'
    DEFAULT_MAX_WAIT_SECONDS = 300
    DEFAULT_POLL_INTERVAL = 2
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        data_access_role_arn: Optional[str] = None
    ):
        """
        Inicializa o serviço Transcribe.
        
        Args:
            region_name: Região AWS (padrão: us-east-1)
            data_access_role_arn: ARN da role para acesso aos dados
        """
        self.region_name = region_name
        self.client = boto3.client('transcribe', region_name=region_name)
        self.data_access_role_arn = data_access_role_arn or (
            'arn:aws:iam::517171444774:role/TranscribeDataAccessRole'
        )
    
    def _validate_media_format(self, s3_path: str) -> Tuple[bool, Optional[str]]:
        """
        Valida o formato de mídia do arquivo.
        
        Args:
            s3_path: Caminho S3 do arquivo
        
        Returns:
            Tupla (válido, mensagem_erro)
        """
        media_format = s3_path.split('.')[-1].lower()
        if media_format not in self.SUPPORTED_FORMATS:
            return False, f"Erro: Formato de áudio '{media_format}' não suportado."
        return True, None
    
    def transcribe(
        self, 
        s3_path: str,
        job_name: Optional[str] = None,
        language_code: str = DEFAULT_LANGUAGE_CODE,
        max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    ) -> str:
        """
        Inicia e recupera transcrição do Amazon Transcribe.
        
        Args:
            s3_path: Caminho S3 do áudio (ex: 's3://bucket/audio.mp3')
            job_name: Nome do job de transcrição (gerado automaticamente se não fornecido)
            language_code: Código do idioma (padrão: pt-BR)
            max_wait_seconds: Tempo máximo de espera em segundos (padrão: 300)
        
        Returns:
            Texto transcrito
        
        Raises:
            ValueError: Se o caminho S3 for inválido
            Exception: Se houver erro na transcrição
        """
        if not s3_path.startswith('s3://'):
            raise ValueError(f"S3 path deve começar com 's3://'. Recebido: {s3_path}")
        
        # Valida formato
        is_valid, error_msg = self._validate_media_format(s3_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Gera nome do job se não fornecido
        if not job_name:
            job_name = f"job_{int(time.time())}"
        
        # Extrai formato de mídia
        media_format = s3_path.split('.')[-1].lower()
        
        # Configura parâmetros do job
        job_params = {
            'TranscriptionJobName': job_name,
            'Media': {'MediaFileUri': s3_path},
            'MediaFormat': media_format,
            'LanguageCode': language_code,
            'Settings': {
                'ShowAlternatives': True,
                'MaxAlternatives': 2,
            },
            'JobExecutionSettings': {
                'DataAccessRoleArn': self.data_access_role_arn,
            }
        }
        
        # Inicia o job de transcrição
        self.client.start_transcription_job(**job_params)
        
        # Aguarda conclusão
        return self._wait_for_completion(job_name, max_wait_seconds)
    
    def _wait_for_completion(
        self, 
        job_name: str, 
        max_wait_seconds: int
    ) -> str:
        """
        Aguarda a conclusão do job de transcrição.
        
        Args:
            job_name: Nome do job
            max_wait_seconds: Tempo máximo de espera
        
        Returns:
            Texto transcrito
        
        Raises:
            TimeoutError: Se o job não completar no tempo máximo
            Exception: Se o job falhar
        """
        elapsed = 0
        
        while elapsed < max_wait_seconds:
            status = self.client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status == 'COMPLETED':
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                return self._fetch_transcript(transcript_uri)
            
            elif job_status == 'FAILED':
                failure_reason = status['TranscriptionJob'].get(
                    'FailureReason', 
                    'Desconhecido'
                )
                raise Exception(f"Falha na transcrição: {failure_reason}")
            
            time.sleep(self.DEFAULT_POLL_INTERVAL)
            elapsed += self.DEFAULT_POLL_INTERVAL
        
        raise TimeoutError(f"Timeout aguardando transcrição após {max_wait_seconds} segundos.")
    
    def _fetch_transcript(self, transcript_uri: str) -> str:
        """
        Busca o texto transcrito a partir da URI.
        
        Args:
            transcript_uri: URI do arquivo de transcrição
        
        Returns:
            Texto transcrito
        """
        with urllib.request.urlopen(transcript_uri) as response:
            transcript_data = json.loads(response.read().decode())
            return transcript_data['results']['transcripts'][0]['transcript']

