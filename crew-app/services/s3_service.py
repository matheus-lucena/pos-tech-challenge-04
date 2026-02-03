"""Serviço para interação com AWS S3."""

import os
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple
from dotenv import load_dotenv
import boto3

# Garante que as variáveis de ambiente estão carregadas
load_dotenv()


class S3Service:
    """Serviço para upload e gerenciamento de arquivos no S3."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Inicializa o serviço S3.
        
        Args:
            region_name: Região AWS (padrão: us-east-1)
        """
        self.region_name = region_name
        self.client = boto3.client('s3', region_name=region_name)
    
    def upload_audio(
        self, 
        file_path: str, 
        bucket_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Faz upload do arquivo de áudio para o S3 e retorna o caminho S3.
        
        Args:
            file_path: Caminho do arquivo local
            bucket_name: Nome do bucket S3 (usa variável de ambiente se não fornecido)
        
        Returns:
            Caminho S3 do arquivo (s3://bucket/key) ou None em caso de erro
        """
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
            
            # Pequeno delay para garantir que o arquivo está disponível
            time.sleep(2)
            
            s3_path = f"s3://{bucket_name}/{s3_key}"
            return s3_path
            
        except Exception as e:
            print(f"Erro ao fazer upload para S3: {str(e)}")
            return None
    
    def verify_file_exists(self, s3_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica se um arquivo existe no S3.
        
        Args:
            s3_path: Caminho S3 (s3://bucket/key)
        
        Returns:
            Tupla (existe, mensagem_erro). Se existe=True, mensagem_erro é None.
        """
        if not s3_path.startswith('s3://'):
            return False, f"Erro: S3 path deve começar com 's3://'. Recebido: {s3_path}"
        
        parts = s3_path.replace('s3://', '').split('/', 1)
        if len(parts) != 2:
            return False, f"Erro: Formato de S3 path inválido. Use: s3://bucket/key. Recebido: {s3_path}"
        
        bucket_name, object_key = parts
        
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_key)
            return True, None
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                return False, f"Erro: Arquivo não encontrado em S3: {s3_path}"
            elif error_code == '403':
                return False, f"Erro: Sem permissão para acessar {s3_path}. Verifique as credenciais AWS."
            else:
                return False, f"Erro ao verificar arquivo S3: {str(e)}"

