"""Tools do CrewAI para análise de saúde materna."""

import json
from dotenv import load_dotenv
from crewai.tools import tool
from services.s3_service import S3Service
from services.sagemaker_service import SageMakerService
from services.transcribe_service import TranscribeService
from services.comprehend_medical_service import ComprehendMedicalService

# Carrega variáveis de ambiente ANTES de criar instâncias dos serviços
load_dotenv()

# Instâncias globais dos serviços para uso nas tools
_s3_service = S3Service()
_sagemaker_service = SageMakerService()
_transcribe_service = TranscribeService()
_comprehend_medical_service = ComprehendMedicalService()


@tool("MaternalRiskPredictor")
def predict_risk(data_json: str) -> str:
    """
    Analisa sinais vitais via SageMaker. Espera JSON com biometria.
    
    Args:
        data_json: String JSON com dados biométricos
    
    Returns:
        String formatada com status e confiança da predição
    """
    try:
        # Converte string JSON para dict
        if isinstance(data_json, str):
            payload = json.loads(data_json.replace("'", '"'))
        else:
            payload = data_json
        
        # Chama o serviço SageMaker
        result = _sagemaker_service.predict_risk(payload)
        
        return (
            f"Status: {result['status']} | "
            f"Confiança: {result['risk_probability']}"
        )
    except Exception as e:
        return f"Erro na predição: {str(e)}"


@tool("AudioTranscriber")
def transcribe_consultation(s3_path: str) -> str:
    """
    Inicia e recupera transcrição do Amazon Transcribe e analisa com Comprehend Medical.
    
    Args:
        s3_path: Caminho S3 do áudio (ex: 's3://bucket/audio.mp3')
    
    Returns:
        Texto transcrito com análise do Comprehend Medical ou mensagem de erro
    """
    # Valida se o arquivo existe no S3
    exists, error_msg = _s3_service.verify_file_exists(s3_path)
    if not exists:
        return error_msg
    
    try:
        # Realiza a transcrição
        transcript = _transcribe_service.transcribe(s3_path)
        
        # Analisa o texto transcrito com Comprehend Medical
        try:
            analysis = _comprehend_medical_service.analyze_text(transcript)
            formatted_analysis = _comprehend_medical_service.format_analysis_result(analysis)
            
            # Retorna transcrição + análise
            result = f"""
=== TRANSCRIÇÃO DO ÁUDIO ===
{transcript}

{formatted_analysis}
"""
            return result
        except Exception as e:
            # Se houver erro no Comprehend Medical, retorna apenas a transcrição
            return f"""
=== TRANSCRIÇÃO DO ÁUDIO ===
{transcript}

⚠️ Aviso: Não foi possível analisar com Comprehend Medical: {str(e)}
"""
        
    except ValueError as e:
        return f"Erro de validação: {str(e)}"
    except TimeoutError as e:
        return f"Erro: {str(e)}"
    except Exception as e:
        return f"Erro na transcrição: {str(e)}"

