"""Serviços externos para integração com AWS."""

from .s3_service import S3Service
from .sagemaker_service import SageMakerService
from .transcribe_service import TranscribeService
from .comprehend_medical_service import ComprehendMedicalService

__all__ = ["S3Service", "SageMakerService", "TranscribeService", "ComprehendMedicalService"]

