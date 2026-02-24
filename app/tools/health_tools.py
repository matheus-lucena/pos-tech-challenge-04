import json
from crewai.tools import tool
from services.instances import (
    get_comprehend_medical_service,
    get_s3_service,
    get_sagemaker_service,
    get_transcribe_service,
)

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
    2. predict_risk() - will use the biometric data set via set_biometric_data()

    Args:
        data_json: Optional JSON string with biometric data. If not provided, uses the cached value.

    Returns:
        Formatted string with status (HIGH RISK or LOW RISK) and confidence of prediction.
    """
    global _biometric_data_cache

    try:
        payload = None
        if data_json is not None:
            if isinstance(data_json, str):
                payload = json.loads(data_json.replace("'", '"'))
            elif isinstance(data_json, dict):
                payload = data_json
        if payload is None and _biometric_data_cache is not None:
            payload = _biometric_data_cache
        if payload is None:
            return (
                "Error: No biometric data provided. "
                "Please provide data_json parameter with biometric data as JSON string."
            )

        required_fields = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
        missing_fields = [f for f in required_fields if f not in payload]
        if missing_fields:
            return f"Error: Missing required fields: {', '.join(missing_fields)}"

        result = get_sagemaker_service().predict_risk(payload)
        return (
            f"Status: {result['status']} | "
            f"Confidence: {result['risk_probability']}"
        )
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format in data_json. {str(e)}"
    except Exception as e:
        return f"Prediction error: {str(e)}"


def set_biometric_data(data: dict):
    global _biometric_data_cache
    _biometric_data_cache = data


@tool("AudioTranscriber")
def transcribe_consultation(s3_path: str) -> str:
    """
    Starts and retrieves transcription from Amazon Transcribe and analyzes with Comprehend Medical.

    Args:
        s3_path: S3 path of audio (e.g., 's3://bucket/audio.mp3')

    Returns:
        Transcribed text with Comprehend Medical analysis or error message.
    """
    exists, error_msg = get_s3_service().verify_file_exists(s3_path)
    if not exists:
        return error_msg

    try:
        transcript = get_transcribe_service().transcribe(s3_path)
        try:
            comprehend = get_comprehend_medical_service()
            analysis = comprehend.analyze_text(transcript)
            formatted_analysis = comprehend.format_analysis_result(analysis)
            return f"\n=== AUDIO TRANSCRIPTION ===\n{transcript}\n\n{formatted_analysis}\n"
        except Exception as e:
            return (
                f"\n=== AUDIO TRANSCRIPTION ===\n{transcript}\n\n"
                f"⚠️ Warning: Could not analyze with Comprehend Medical: {str(e)}\n"
            )

    except ValueError as e:
        return f"Validation error: {str(e)}"
    except TimeoutError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Transcription error: {str(e)}"
