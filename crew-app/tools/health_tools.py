import json
from dotenv import load_dotenv
from crewai.tools import tool
from services.s3_service import S3Service
from services.sagemaker_service import SageMakerService
from services.transcribe_service import TranscribeService
from services.comprehend_medical_service import ComprehendMedicalService

load_dotenv()

_s3_service = S3Service()
_sagemaker_service = SageMakerService()
_transcribe_service = TranscribeService()
_comprehend_medical_service = ComprehendMedicalService()


@tool("MaternalRiskPredictor")
def predict_risk(data_json: str) -> str:
    """
    Analyzes vital signs via SageMaker. Expects JSON with biometric data.
    
    Args:
        data_json: JSON string with biometric data
    
    Returns:
        Formatted string with status and confidence of prediction
    """
    try:
        if isinstance(data_json, str):
            payload = json.loads(data_json.replace("'", '"'))
        else:
            payload = data_json
        
        result = _sagemaker_service.predict_risk(payload)
        
        return (
            f"Status: {result['status']} | "
            f"Confidence: {result['risk_probability']}"
        )
    except Exception as e:
        return f"Prediction error: {str(e)}"


@tool("AudioTranscriber")
def transcribe_consultation(s3_path: str) -> str:
    """
    Starts and retrieves transcription from Amazon Transcribe and analyzes with Comprehend Medical.
    
    Args:
        s3_path: S3 path of audio (e.g., 's3://bucket/audio.mp3')
    
    Returns:
        Transcribed text with Comprehend Medical analysis or error message
    """
    exists, error_msg = _s3_service.verify_file_exists(s3_path)
    if not exists:
        return error_msg
    
    try:
        transcript = _transcribe_service.transcribe(s3_path)
        
        try:
            analysis = _comprehend_medical_service.analyze_text(transcript)
            formatted_analysis = _comprehend_medical_service.format_analysis_result(analysis)
            
            result = f"""
=== AUDIO TRANSCRIPTION ===
{transcript}

{formatted_analysis}
"""
            return result
        except Exception as e:
            return f"""
=== AUDIO TRANSCRIPTION ===
{transcript}

⚠️ Warning: Could not analyze with Comprehend Medical: {str(e)}
"""
        
    except ValueError as e:
        return f"Validation error: {str(e)}"
    except TimeoutError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Transcription error: {str(e)}"
