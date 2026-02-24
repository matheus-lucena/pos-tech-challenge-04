import os
import tempfile

import librosa
from crewai.tools import tool

from services.instances import get_maternal_health_service, get_s3_service

_maternal_audio_cache: str | None = None


def set_maternal_audio_path(path: str) -> None:
    global _maternal_audio_cache
    _maternal_audio_cache = path


@tool("MaternalHeartSoundAnalyzer")
def analyze_maternal_heart_sound(
    audio_path: str = None,
    is_s3_path: bool = False,
    **kwargs,
) -> str:
    """
    Analyzes maternal heart signals (PCG) from an audio file.

    Args:
        audio_path: Local file path or S3 URI (s3://bucket/key).
                    If omitted, uses the path set via set_maternal_audio_path().
        is_s3_path: Set to True when audio_path is an S3 URI.
    """
    global _maternal_audio_cache
    if audio_path is None:
        audio_path = _maternal_audio_cache
    if audio_path is None:
        return (
            "Error: No audio path provided. "
            "Pass audio_path='s3://bucket/key' or call set_maternal_audio_path() first."
        )
    if audio_path.startswith("s3://"):
        is_s3_path = True
    try:
        local_path = audio_path
        temp_file = None

        if is_s3_path:
            exists, error_msg = get_s3_service().verify_file_exists(audio_path)
            if not exists:
                return f"Error: {error_msg}"

            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.close()
                get_s3_service().download_file(audio_path, temp_file.name)
                local_path = temp_file.name
            except Exception as e:
                return f"Error downloading file from S3: {str(e)}"

        result = get_maternal_health_service().analyze_maternal_signal(local_path)

        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

        if result.get("status") == "error":
            return f"❌ Analysis error: {result.get('error', 'Unknown error')}"

        output = f"""
=== MATERNAL SIGNAL ANALYSIS ===

📊 MATERNAL HEART RATE (MHR):
   • MHR: {result['maternal_heart_rate']} bpm
   • Confidence: {result['mhr_confidence']*100:.1f}%
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
        for rec in result["recommendations"]:
            output += f"   • {rec}\n"

        return output.strip()

    except Exception as e:
        return f"Error analyzing maternal signal: {str(e)}"
