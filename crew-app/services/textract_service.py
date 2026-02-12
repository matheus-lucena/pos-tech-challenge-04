import os
import json
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import boto3

load_dotenv()


class TextractService:
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client('textract', region_name=self.region_name)
        self.s3_client = boto3.client('s3', region_name=self.region_name)
    
    def extract_text_from_pdf_s3(
        self, 
        s3_path: str,
        bucket_name: Optional[str] = None
    ) -> Optional[str]:
        if not s3_path.startswith('s3://'):
            return None
        
        parts = s3_path.replace('s3://', '').split('/', 1)
        if len(parts) != 2:
            return None
        
        bucket_name, object_key = parts
        
        try:
            response = self.client.detect_document_text(
                Document={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': object_key
                    }
                }
            )
            
            text_blocks = []
            for block in response.get('Blocks', []):
                if block['BlockType'] == 'LINE':
                    text_blocks.append(block.get('Text', ''))
            
            return '\n'.join(text_blocks)
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def extract_text_from_pdf_local(
        self, 
        file_path: str,
        upload_to_s3: bool = True
    ) -> Optional[str]:
        if not os.path.exists(file_path):
            return None
        
        if upload_to_s3:
            s3_path = self._upload_pdf_to_s3(file_path)
            if not s3_path:
                return None
            return self.extract_text_from_pdf_s3(s3_path)
        else:
            try:
                with open(file_path, 'rb') as document:
                    response = self.client.detect_document_text(
                        Document={'Bytes': document.read()}
                    )
                
                text_blocks = []
                for block in response.get('Blocks', []):
                    if block['BlockType'] == 'LINE':
                        text_blocks.append(block.get('Text', ''))
                
                return '\n'.join(text_blocks)
            except Exception as e:
                print(f"Error extracting text from local PDF: {str(e)}")
                return None
    
    def _upload_pdf_to_s3(self, file_path: str) -> Optional[str]:
        from datetime import datetime
        import uuid
        
        bucket_name = os.getenv("S3_BUCKET_NAME", "fiap-pos-teste-20-19")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"pdf-uploads/{timestamp}_{uuid.uuid4().hex[:8]}.pdf"
            
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'application/pdf'}
            )
            
            time.sleep(2)
            return f"s3://{bucket_name}/{s3_key}"
            
        except Exception as e:
            print(f"Error uploading PDF to S3: {str(e)}")
            return None
