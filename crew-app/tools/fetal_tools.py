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
    Analyzes fetal heart signals (PCG) from an audio file.
    Extracts fetal heart rate (FHR), variability, and classifies the state.
    
    Args:
        audio_path: Path to audio file (local or S3 if is_s3_path=True)
        is_s3_path: If True, audio_path is an S3 path (s3://bucket/key)
    
    Returns:
        Formatted string with fetal analysis results
    """
    try:
        local_path = audio_path
        temp_file = None
        
        if is_s3_path:
            if not audio_path.startswith('s3://'):
                return f"Error: Invalid S3 path. Must start with 's3://': {audio_path}"
            
            exists, error_msg = _s3_service.verify_file_exists(audio_path)
            if not exists:
                return f"Error: {error_msg}"
            
            try:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_file.close()
                
                _s3_service.download_file(audio_path, temp_file.name)
                local_path = temp_file.name
            except Exception as e:
                return f"Error downloading file from S3: {str(e)}"
        
        result = _fetal_service.analyze_fetal_signal(local_path)
        
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        if result.get("status") == "error":
            return f"❌ Analysis error: {result.get('error', 'Unknown error')}"
        
        output = f"""
=== FETAL SIGNAL ANALYSIS ===

📊 FETAL HEART RATE (FHR):
   • FHR: {result['fetal_heart_rate']} bpm
   • Confidence: {result['fhr_confidence']*100:.1f}%
   • Variability: {result['variability']:.2f} bpm

🏥 CLASSIFICATION:
   • Status: {result['classification']['status'].upper()}
   • Risk Level: {result['classification']['risk_level'].upper()}
   • Variability: {result['classification']['variability_status'].upper()}
   • Description: {result['classification']['description']}

📈 SIGNAL QUALITY:
   • Quality: {result['signal_quality'].upper()}
   • Beats Detected: {result['num_beats_detected']}

🔬 SPECTRAL FEATURES:
   • Spectral Centroid: {result['spectral_features']['spectral_centroid']} Hz
   • Bandwidth: {result['spectral_features']['spectral_bandwidth']:.2f} Hz
   • Low Energy (20-100 Hz): {result['spectral_features']['energy_low_band']*100:.1f}%
   • Mid Energy (100-300 Hz): {result['spectral_features']['energy_mid_band']*100:.1f}%
   • High Energy (300-500 Hz): {result['spectral_features']['energy_high_band']*100:.1f}%

💡 RECOMMENDATIONS:
"""
        for rec in result['recommendations']:
            output += f"   • {rec}\n"
        
        return output.strip()
        
    except Exception as e:
        return f"Error analyzing fetal signal: {str(e)}"


@tool("FetalRealtimeAnalyzer")
def analyze_fetal_realtime(
    audio_chunk_path: str,
    sample_rate: int = 16000
) -> str:
    """
    Analyzes an audio chunk in real-time for continuous monitoring.
    Useful for streaming analysis or real-time processing.
    
    Args:
        audio_chunk_path: Path to audio chunk
        sample_rate: Sample rate (default: 16000 Hz)
    
    Returns:
        Formatted string with real-time analysis results
    """
    try:
        import numpy as np
        import librosa
        
        y, sr = librosa.load(audio_chunk_path, sr=sample_rate, mono=True)
        
        result = _fetal_service.analyze_realtime_stream(y, sr)
        
        if result.get("status") == "error":
            return f"❌ Error: {result.get('error', 'Unknown error')}"
        
        output = (
            f"FHR: {result['fetal_heart_rate']:.1f} bpm | "
            f"Risk: {result['classification']['risk_level'].upper()} | "
            f"Status: {result['classification']['status'].upper()}"
        )
        
        return output
        
    except Exception as e:
        return f"Real-time analysis error: {str(e)}"
