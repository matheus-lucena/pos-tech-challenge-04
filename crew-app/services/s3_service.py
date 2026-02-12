import os
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple
from dotenv import load_dotenv
import boto3

load_dotenv()


class S3Service:
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.client = boto3.client('s3', region_name=region_name)
    
    def upload_audio(
        self, 
        file_path: str, 
        bucket_name: Optional[str] = None
    ) -> Optional[str]:
        if not file_path or not os.path.exists(file_path):
            return None
        
        if not bucket_name:
            bucket_name = os.getenv("S3_BUCKET_NAME", "fiap-pos-teste-20-19")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(file_path)[1]
            s3_key = f"audio-uploads/{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
            
            content_type = 'audio/mpeg' if file_extension == '.mp3' else 'audio/wav'
            
            self.client.upload_file(
                file_path,
                bucket_name,
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )
            
            time.sleep(2)
            return f"s3://{bucket_name}/{s3_key}"
            
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
            return None
    
    def verify_file_exists(self, s3_path: str) -> Tuple[bool, Optional[str]]:
        if not s3_path.startswith('s3://'):
            return False, f"Error: S3 path must start with 's3://'. Received: {s3_path}"
        
        parts = s3_path.replace('s3://', '').split('/', 1)
        if len(parts) != 2:
            return False, f"Error: Invalid S3 path format. Use: s3://bucket/key. Received: {s3_path}"
        
        bucket_name, object_key = parts
        
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_key)
            return True, None
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                return False, f"Error: File not found in S3: {s3_path}"
            elif error_code == '403':
                return False, f"Error: No permission to access {s3_path}. Check AWS credentials."
            else:
                return False, f"Error verifying S3 file: {str(e)}"
    
    def download_file(self, s3_path: str, local_path: str) -> bool:
        if not s3_path.startswith('s3://'):
            return False
        
        parts = s3_path.replace('s3://', '').split('/', 1)
        if len(parts) != 2:
            return False
        
        bucket_name, object_key = parts
        
        try:
            os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else '.', exist_ok=True)
            self.client.download_file(bucket_name, object_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading file from S3: {str(e)}")
            return False
