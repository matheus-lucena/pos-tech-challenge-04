import os
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple

import boto3

from config.constants import S3_UPLOAD_PROPAGATION_WAIT
from utils.s3_utils import parse_s3_path

_AUDIO_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".amr": "audio/amr",
}


class S3Service:
    def __init__(self, region_name: Optional[str] = None):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("s3", region_name=self.region_name)

    def upload_audio(
        self,
        file_path: str,
        bucket_name: Optional[str] = None,
    ) -> Optional[str]:
        if not file_path or not os.path.exists(file_path):
            return None

        if not bucket_name:
            bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3 bucket name not configured. "
                "Set the S3_BUCKET_NAME environment variable or pass bucket_name to upload_audio()."
            )

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(file_path)[1].lower()
            s3_key = f"audio-uploads/{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
            content_type = _AUDIO_CONTENT_TYPES.get(file_extension, "application/octet-stream")

            self.client.upload_file(
                file_path,
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )

            time.sleep(S3_UPLOAD_PROPAGATION_WAIT)
            return f"s3://{bucket_name}/{s3_key}"

        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
            return None

    def verify_file_exists(self, s3_path: str) -> Tuple[bool, Optional[str]]:
        if not s3_path.startswith("s3://"):
            return False, f"Error: S3 path must start with 's3://'. Received: {s3_path}"

        try:
            bucket_name, object_key = parse_s3_path(s3_path)
        except ValueError as e:
            return False, f"Error: {str(e)}"

        try:
            self.client.head_object(Bucket=bucket_name, Key=object_key)
            return True, None
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False, f"Error: File not found in S3: {s3_path}"
            elif error_code == "403":
                return False, f"Error: No permission to access {s3_path}. Check AWS credentials."
            else:
                return False, f"Error verifying S3 file: {str(e)}"

    def download_file(self, s3_path: str, local_path: str) -> bool:
        if not s3_path.startswith("s3://"):
            return False

        try:
            bucket_name, object_key = parse_s3_path(s3_path)
        except ValueError:
            return False

        try:
            os.makedirs(
                os.path.dirname(local_path) if os.path.dirname(local_path) else ".",
                exist_ok=True,
            )
            self.client.download_file(bucket_name, object_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading file from S3: {str(e)}")
            return False
