from typing import Optional

from .comprehend_medical_service import ComprehendMedicalService
from .maternal_health_service import MaternalHealthService
from .pdf_parser_service import PDFParserService
from .s3_service import S3Service
from .sagemaker_service import SageMakerService
from .textract_service import TextractService
from .transcribe_service import TranscribeService
from .transcribe_streaming_service import TranscribeStreamingService

_s3_service: Optional[S3Service] = None
_sagemaker_service: Optional[SageMakerService] = None
_transcribe_service: Optional[TranscribeService] = None
_comprehend_medical_service: Optional[ComprehendMedicalService] = None
_maternal_health_service: Optional[MaternalHealthService] = None
_textract_service: Optional[TextractService] = None
_transcribe_streaming_service: Optional[TranscribeStreamingService] = None
_pdf_parser_service: Optional[PDFParserService] = None


def get_s3_service() -> S3Service:
    global _s3_service
    if _s3_service is None:
        _s3_service = S3Service()
    return _s3_service


def get_sagemaker_service() -> SageMakerService:
    global _sagemaker_service
    if _sagemaker_service is None:
        _sagemaker_service = SageMakerService()
    return _sagemaker_service


def get_transcribe_service() -> TranscribeService:
    global _transcribe_service
    if _transcribe_service is None:
        _transcribe_service = TranscribeService()
    return _transcribe_service


def get_comprehend_medical_service() -> ComprehendMedicalService:
    global _comprehend_medical_service
    if _comprehend_medical_service is None:
        _comprehend_medical_service = ComprehendMedicalService()
    return _comprehend_medical_service


def get_maternal_health_service() -> MaternalHealthService:
    global _maternal_health_service
    if _maternal_health_service is None:
        _maternal_health_service = MaternalHealthService()
    return _maternal_health_service


def get_textract_service() -> TextractService:
    global _textract_service
    if _textract_service is None:
        _textract_service = TextractService()
    return _textract_service


def get_transcribe_streaming_service() -> TranscribeStreamingService:
    global _transcribe_streaming_service
    if _transcribe_streaming_service is None:
        _transcribe_streaming_service = TranscribeStreamingService()
    return _transcribe_streaming_service


def get_pdf_parser_service() -> PDFParserService:
    global _pdf_parser_service
    if _pdf_parser_service is None:
        _pdf_parser_service = PDFParserService(
            textract_service=get_textract_service(),
            comprehend_service=get_comprehend_medical_service(),
        )
    return _pdf_parser_service
