import json
import os
import time
import urllib.request
from typing import Optional, Tuple
from dotenv import load_dotenv
import boto3

load_dotenv()


class TranscribeService:
    SUPPORTED_FORMATS = ['mp3', 'mp4', 'wav', 'flac', 'ogg', 'amr', 'webm']
    DEFAULT_LANGUAGE_CODE = 'pt-BR'
    DEFAULT_MAX_WAIT_SECONDS = 300
    DEFAULT_POLL_INTERVAL = 2
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        data_access_role_arn: Optional[str] = None
    ):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client('transcribe', region_name=self.region_name)
        self.data_access_role_arn = data_access_role_arn or os.getenv("AWS_TRANSCRIBE_ROLE_ARN")
        if not self.data_access_role_arn:
            raise ValueError(
                "TRANSCRIBE_ROLE_ARN not configured. "
                "Set AWS_TRANSCRIBE_ROLE_ARN environment variable or pass data_access_role_arn to constructor."
            )

    def _validate_media_format(self, s3_path: str) -> Tuple[bool, Optional[str]]:
        media_format = s3_path.split('.')[-1].lower()
        if media_format not in self.SUPPORTED_FORMATS:
            return False, f"Error: Audio format '{media_format}' not supported."
        return True, None
    
    def transcribe(
        self, 
        s3_path: str,
        job_name: Optional[str] = None,
        language_code: str = DEFAULT_LANGUAGE_CODE,
        max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    ) -> str:
        if not s3_path.startswith('s3://'):
            raise ValueError(f"S3 path must start with 's3://'. Received: {s3_path}")
        
        is_valid, error_msg = self._validate_media_format(s3_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        if not job_name:
            job_name = f"job_{int(time.time())}"
        
        media_format = s3_path.split('.')[-1].lower()
        
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
        
        self.client.start_transcription_job(**job_params)
        return self._wait_for_completion(job_name, max_wait_seconds)
    
    def _wait_for_completion(
        self, 
        job_name: str, 
        max_wait_seconds: int
    ) -> str:
        elapsed = 0
        
        while elapsed < max_wait_seconds:
            status = self.client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status == 'COMPLETED':
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                return self._fetch_transcript(transcript_uri)
            
            elif job_status == 'FAILED':
                failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown')
                raise Exception(f"Transcription failed: {failure_reason}")
            
            time.sleep(self.DEFAULT_POLL_INTERVAL)
            elapsed += self.DEFAULT_POLL_INTERVAL
        
        raise TimeoutError(f"Timeout waiting for transcription after {max_wait_seconds} seconds.")
    
    def _fetch_transcript(self, transcript_uri: str) -> str:
        with urllib.request.urlopen(transcript_uri) as response:
            transcript_data = json.loads(response.read().decode())
            return transcript_data['results']['transcripts'][0]['transcript']
