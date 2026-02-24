import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import boto3

from config.constants import S3_UPLOAD_PROPAGATION_WAIT
from utils.s3_utils import parse_s3_path


class TextractService:
    def __init__(self, region_name: Optional[str] = None):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("textract", region_name=self.region_name)
        self.s3_client = boto3.client("s3", region_name=self.region_name)

    def extract_text_from_pdf_s3(self, s3_path: str) -> Optional[str]:
        if not s3_path.startswith("s3://"):
            return None

        try:
            bucket_name, object_key = parse_s3_path(s3_path)
        except ValueError:
            return None

        try:
            response = self.client.detect_document_text(
                Document={"S3Object": {"Bucket": bucket_name, "Name": object_key}}
            )
            return self._extract_text_blocks(response)
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None

    def extract_text_from_pdf_local(self, file_path: str) -> Optional[str]:
        """Extract text from a local PDF file.

        Sends the file bytes directly to Textract (no S3 required).
        If Textract rejects the bytes (e.g. multi-page PDF not supported via bytes),
        it falls back to uploading the file to S3 first.
        """
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as document:
                response = self.client.detect_document_text(
                    Document={"Bytes": document.read()}
                )
            return self._extract_text_blocks(response)
        except Exception as bytes_err:
            print(
                f"Textract bytes extraction failed ({bytes_err}). "
                "Retrying via S3 upload..."
            )

        s3_path = self._upload_pdf_to_s3(file_path)
        if not s3_path:
            return None
        return self.extract_text_from_pdf_s3(s3_path)

    @staticmethod
    def _extract_text_blocks(response: Dict[str, Any]) -> str:
        """Return the plain text content from a Textract DetectDocumentText response."""
        lines = [
            block.get("Text", "")
            for block in response.get("Blocks", [])
            if block["BlockType"] == "LINE"
        ]
        return "\n".join(lines)

    def _upload_pdf_to_s3(self, file_path: str) -> Optional[str]:
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3 bucket name not configured. "
                "Set the S3_BUCKET_NAME environment variable."
            )

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"pdf-uploads/{timestamp}_{uuid.uuid4().hex[:8]}.pdf"

            self.s3_client.upload_file(
                file_path,
                bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "application/pdf"},
            )

            time.sleep(S3_UPLOAD_PROPAGATION_WAIT)
            return f"s3://{bucket_name}/{s3_key}"

        except Exception as e:
            print(f"Error uploading PDF to S3: {str(e)}")
            return None
