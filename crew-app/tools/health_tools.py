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


# Variável global para armazenar dados biométricos temporariamente
_biometric_data_cache = None

@tool("MaternalRiskPredictor")
def predict_risk(data_json: str = None, **kwargs) -> str:
    """
    Analyzes vital signs via SageMaker. Expects JSON with biometric data.
    
    This tool analyzes maternal health risk based on biometric data including:
    - Age
    - SystolicBP (Systolic Blood Pressure)
    - DiastolicBP (Diastolic Blood Pressure)
    - BS (Blood Sugar/Glucose)
    - BodyTemp (Body Temperature)
    - HeartRate
    
    You can call this tool in two ways:
    1. predict_risk(data_json='{"Age": 35, "SystolicBP": 140, "DiastolicBP": 90, "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 70}')
    2. predict_risk() - will use the biometric data from the task description
    
    Args:
        data_json: Optional - JSON string with biometric data. If not provided, will try to extract from context.
    
    Returns:
        Formatted string with status (HIGH RISK or LOW RISK) and confidence of prediction
    """
    global _biometric_data_cache
    
    try:
        payload = None
        
        # Tenta obter dados do parâmetro data_json
        if data_json is not None:
            if isinstance(data_json, str):
                payload = json.loads(data_json.replace("'", '"'))
            elif isinstance(data_json, dict):
                payload = data_json
        
        # Se não tem payload e tem cache, usa o cache
        if payload is None and _biometric_data_cache is not None:
            payload = _biometric_data_cache
        
        # Se ainda não tem payload, retorna erro
        if payload is None:
            return "Error: No biometric data provided. Please provide data_json parameter with biometric data as JSON string."
        
        required_fields = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            return f"Error: Missing required fields: {', '.join(missing_fields)}"
    
        result = _sagemaker_service.predict_risk(payload)
        
        return (
            f"Status: {result['status']} | "
            f"Confidence: {result['risk_probability']}"
        )
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format in data_json. {str(e)}"
    except Exception as e:
        return f"Prediction error: {str(e)}"


def set_biometric_data(data: dict):
    """Helper function to set biometric data cache."""
    global _biometric_data_cache
    _biometric_data_cache = data


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
