"""Serviço para interação com AWS SageMaker."""

import os
import json
from typing import Dict, Any, Optional
import boto3


class SageMakerService:
    """Serviço para predições usando modelos SageMaker."""
    
    def __init__(
        self, 
        endpoint_name: Optional[str] = None,
        region_name: str = "us-east-1"
    ):
        """
        Inicializa o serviço SageMaker.
        
        Args:
            endpoint_name: Nome do endpoint (usa variável de ambiente se não fornecido)
            region_name: Região AWS (padrão: us-east-1)
        """
        self.endpoint_name = endpoint_name or os.getenv(
            "SAGEMAKER_ENDPOINT", 
            "DEFAULT_SAGEMAKER_ENDPOINT"
        )
        self.region_name = region_name
        self.client = boto3.client("sagemaker-runtime", region_name=region_name)
    
    def predict_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa sinais vitais via SageMaker.
        
        Args:
            data: Dicionário com dados biométricos (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate)
        
        Returns:
            Dicionário com status e probabilidade de risco
        """
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(data)
            )
            
            result = json.loads(response['Body'].read().decode())
            status = "ALTO RISCO" if result.get("maternal_health_risk") == 1 else "BAIXO RISCO"
            
            return {
                "status": status,
                "risk_probability": result.get('risk_probability', 0),
                "raw_result": result
            }
        except Exception as e:
            raise Exception(f"Erro na predição SageMaker: {str(e)}")

