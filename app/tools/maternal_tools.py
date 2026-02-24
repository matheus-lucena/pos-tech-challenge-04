import os
import tempfile

import librosa
from crewai.tools import tool

from services.instances import get_maternal_health_service, get_s3_service


@tool("MaternalHeartSoundAnalyzer")
def analyze_maternal_heart_sound(
    audio_path: str,
    is_s3_path: bool = False,
) -> str:
    """Analyzes maternal heart signals (PCG) from an audio file. Use is_s3_path=True if audio_path is s3://bucket/key."""
    try:
        local_path = audio_path
        temp_file = None

        if is_s3_path:
            if not audio_path.startswith("s3://"):
                return f"Error: Invalid S3 path. Must start with 's3://': {audio_path}"

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
