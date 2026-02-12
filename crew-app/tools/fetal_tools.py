"""Tools do CrewAI para análise de sinais fetais."""

import json
from typing import Optional
from dotenv import load_dotenv
from crewai.tools import tool
from services.fetal_analysis_service import FetalAnalysisService
from services.s3_service import S3Service
import os

load_dotenv()

_fetal_service = FetalAnalysisService()
_s3_service = S3Service()


@tool("FetalHeartSoundAnalyzer")
def analyze_fetal_heart_sound(
    audio_path: str,
    is_s3_path: bool = False
) -> str:
    """
    Analisa sinais de coração fetal (PCG) de um arquivo de áudio.
    Extrai frequência cardíaca fetal (FHR), variabilidade, e classifica o estado.
    
    Args:
        audio_path: Caminho para o arquivo de áudio (local ou S3 se is_s3_path=True)
        is_s3_path: Se True, audio_path é um caminho S3 (s3://bucket/key)
    
    Returns:
        String formatada com resultados da análise fetal
    """
    try:
        local_path = audio_path
        temp_file = None
        
        if is_s3_path:
            if not audio_path.startswith('s3://'):
                return f"Erro: Caminho S3 inválido. Deve começar com 's3://': {audio_path}"
            
            exists, error_msg = _s3_service.verify_file_exists(audio_path)
            if not exists:
                return f"Erro: {error_msg}"
            
            try:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_file.close()
                
                _s3_service.download_file(audio_path, temp_file.name)
                local_path = temp_file.name
            except Exception as e:
                return f"Erro ao baixar arquivo do S3: {str(e)}"
        
        result = _fetal_service.analyze_fetal_signal(local_path)
        
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        if result.get("status") == "error":
            return f"❌ Erro na análise: {result.get('error', 'Erro desconhecido')}"
        
        output = f"""
=== ANÁLISE DE SINAL FETAL ===

📊 FREQUÊNCIA CARDÍACA FETAL (FHR):
   • FHR: {result['fetal_heart_rate']} bpm
   • Confiança: {result['fhr_confidence']*100:.1f}%
   • Variabilidade: {result['variability']:.2f} bpm

🏥 CLASSIFICAÇÃO:
   • Status: {result['classification']['status'].upper()}
   • Nível de Risco: {result['classification']['risk_level'].upper()}
   • Variabilidade: {result['classification']['variability_status'].upper()}
   • Descrição: {result['classification']['description']}

📈 QUALIDADE DO SINAL:
   • Qualidade: {result['signal_quality'].upper()}
   • Batimentos Detectados: {result['num_beats_detected']}

🔬 CARACTERÍSTICAS ESPECTRAIS:
   • Centróide Espectral: {result['spectral_features']['spectral_centroid']} Hz
   • Largura de Banda: {result['spectral_features']['spectral_bandwidth']:.2f} Hz
   • Energia Baixa (20-100 Hz): {result['spectral_features']['energy_low_band']*100:.1f}%
   • Energia Média (100-300 Hz): {result['spectral_features']['energy_mid_band']*100:.1f}%
   • Energia Alta (300-500 Hz): {result['spectral_features']['energy_high_band']*100:.1f}%

💡 RECOMENDAÇÕES:
"""
        for rec in result['recommendations']:
            output += f"   • {rec}\n"
        
        return output.strip()
        
    except Exception as e:
        return f"Erro ao analisar sinal fetal: {str(e)}"


@tool("FetalRealtimeAnalyzer")
def analyze_fetal_realtime(
    audio_chunk_path: str,
    sample_rate: int = 16000
) -> str:
    """
    Analisa um chunk de áudio em tempo real para monitoramento contínuo.
    Útil para análise de streaming ou processamento em tempo real.
    
    Args:
        audio_chunk_path: Caminho para o chunk de áudio
        sample_rate: Taxa de amostragem (padrão: 16000 Hz)
    
    Returns:
        String formatada com resultados da análise em tempo real
    """
    try:
        import numpy as np
        import librosa
        
        y, sr = librosa.load(audio_chunk_path, sr=sample_rate, mono=True)
        
        result = _fetal_service.analyze_realtime_stream(y, sr)
        
        if result.get("status") == "error":
            return f"❌ Erro: {result.get('error', 'Erro desconhecido')}"
        
        output = (
            f"FHR: {result['fetal_heart_rate']:.1f} bpm | "
            f"Risco: {result['classification']['risk_level'].upper()} | "
            f"Status: {result['classification']['status'].upper()}"
        )
        
        return output
        
    except Exception as e:
        return f"Erro na análise em tempo real: {str(e)}"

