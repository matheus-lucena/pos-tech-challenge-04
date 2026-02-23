import warnings
from typing import Dict, Any, Optional, Tuple

import librosa
import numpy as np
import scipy.signal

from config.constants import MHR_VALID_MIN, MHR_VALID_MAX, SAMPLE_RATE

warnings.filterwarnings('ignore')


class MaternalHealthService:
    MHR_NORMAL_MIN = 60
    MHR_NORMAL_MAX = 110
    MHR_TACHYCARDIA = 110
    MHR_BRADYCARDIA = 60

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.max_analysis_seconds = 30

    def analyze_maternal_signal(
        self,
        audio_path: str,
        sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate

            try:
                y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            except Exception as e:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                if sr != sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                    sr = sample_rate

            max_len = int(sr * self.max_analysis_seconds)
            if max_len > 0 and len(y) > max_len:
                y = y[:max_len]

            return self._process_signal(y, sr)

        except Exception as e:
            return {
                "error": f"Error processing signal: {str(e)}",
                "status": "error"
            }

    def _process_signal(self, signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        sos = scipy.signal.butter(4, [20, 500], btype='band', fs=sample_rate, output='sos')
        filtered_signal = scipy.signal.sosfilt(sos, signal)

        filtered_signal = filtered_signal - np.mean(filtered_signal)

        if np.max(np.abs(filtered_signal)) > 0:
            filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))

        beats = self._detect_heartbeats(filtered_signal, sample_rate)
        mhr, mhr_confidence = self._estimate_mhr_from_beats(beats, sample_rate)
        if mhr == 0.0:
            mhr, mhr_confidence = self._estimate_mhr_fft(filtered_signal, sample_rate)
        variability = self._calculate_variability(beats, sample_rate)
        classification = self._classify_mhr(mhr, variability)
        spectral_features = self._extract_spectral_features(filtered_signal, sample_rate)

        return {
            "maternal_heart_rate": round(mhr, 2),
            "mhr_confidence": round(mhr_confidence, 2),
            "variability": round(variability, 2),
            "classification": classification,
            "num_beats_detected": len(beats),
            "signal_quality": self._assess_signal_quality(filtered_signal),
            "spectral_features": spectral_features,
            "status": "success",
            "recommendations": self._generate_recommendations(mhr, variability, classification)
        }

    def _estimate_mhr_from_beats(
        self,
        beats: np.ndarray,
        sample_rate: int
    ) -> Tuple[float, float]:
        if beats is None or len(beats) < 2:
            return 0.0, 0.0

        rr_intervals = np.diff(beats) / sample_rate
        rr_intervals = rr_intervals[rr_intervals > 0]
        if len(rr_intervals) == 0:
            return 0.0, 0.0

        rr_median = float(np.median(rr_intervals))
        mhr = 60.0 / rr_median if rr_median > 0 else 0.0

        rr_std = float(np.std(rr_intervals)) if len(rr_intervals) > 1 else 0.0
        cv = (rr_std / rr_median) if rr_median > 0 else 1.0
        confidence = float(max(0.0, min(1.0, 1.0 - cv)))

        if mhr < MHR_VALID_MIN or mhr > MHR_VALID_MAX:
            return 0.0, 0.0
        return mhr, confidence

    def _estimate_mhr_fft(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> Tuple[float, float]:
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft)

        freqs_bpm = freqs * 60

        mask = (freqs_bpm >= self.MHR_BRADYCARDIA) & (freqs_bpm <= self.MHR_TACHYCARDIA)
        if not np.any(mask):
            return 0.0, 0.0

        search_freqs = freqs_bpm[mask]
        search_magnitude = magnitude[mask]

        peak_idx = np.argmax(search_magnitude)
        mhr = search_freqs[peak_idx]

        max_magnitude = np.max(magnitude)
        confidence = search_magnitude[peak_idx] / max_magnitude if max_magnitude > 0 else 0

        return mhr, confidence

    def _detect_heartbeats(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        sos = scipy.signal.butter(4, [50, 300], btype='band', fs=sample_rate, output='sos')
        filtered = scipy.signal.sosfilt(sos, signal)

        min_distance = int(sample_rate * 60 / self.MHR_TACHYCARDIA)

        peaks, _ = scipy.signal.find_peaks(
            np.abs(filtered),
            distance=min_distance,
            prominence=np.std(filtered) * 0.5
        )

        return peaks

    def _calculate_variability(
        self,
        beats: np.ndarray,
        sample_rate: int
    ) -> float:
        if len(beats) < 2:
            return 0.0

        rr_intervals = np.diff(beats) / sample_rate
        rr_intervals_bpm = 60 / rr_intervals
        variability = np.std(rr_intervals_bpm) if len(rr_intervals_bpm) > 1 else 0.0
        return variability

    def _classify_mhr(
        self,
        mhr: float,
        variability: float
    ) -> Dict[str, Any]:
        if mhr == 0:
            return {
                "status": "indeterminado",
                "risk_level": "unknown",
                "description": "Could not determine MHR"
            }

        if mhr < self.MHR_BRADYCARDIA:
            mhr_status = "bradicardia"
            risk_level = "moderado"
            description = f"MHR abaixo do esperado em repouso na gestação ({mhr:.1f} bpm)."
        elif mhr > self.MHR_TACHYCARDIA:
            mhr_status = "taquicardia"
            risk_level = "moderado"
            description = f"MHR acima do esperado em repouso na gestação ({mhr:.1f} bpm)."
        else:
            mhr_status = "normal"
            risk_level = "baixo"
            description = f"MHR dentro da faixa esperada em repouso na gestação ({mhr:.1f} bpm)."

        if variability < 5:
            variability_status = "baixa"
            if risk_level == "baixo":
                risk_level = "moderado"
            description += " Variabilidade baixa detectada."
        elif variability > 25:
            variability_status = "alta"
            description += " Variabilidade alta (pode ocorrer)."
        else:
            variability_status = "normal"

        return {
            "status": mhr_status,
            "variability_status": variability_status,
            "risk_level": risk_level,
            "description": description
        }

    def _extract_spectral_features(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        freqs, psd = scipy.signal.welch(signal, sample_rate, nperseg=min(2048, len(signal)))

        centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0

        low_band = (freqs >= 20) & (freqs <= 100)
        mid_band = (freqs > 100) & (freqs <= 300)
        high_band = (freqs > 300) & (freqs <= 500)

        energy_low = np.sum(psd[low_band]) if np.any(low_band) else 0
        energy_mid = np.sum(psd[mid_band]) if np.any(mid_band) else 0
        energy_high = np.sum(psd[high_band]) if np.any(high_band) else 0
        total_energy = np.sum(psd)

        return {
            "spectral_centroid": round(centroid, 2),
            "spectral_bandwidth": round(bandwidth, 2),
            "energy_low_band": round(energy_low / total_energy if total_energy > 0 else 0, 3),
            "energy_mid_band": round(energy_mid / total_energy if total_energy > 0 else 0, 3),
            "energy_high_band": round(energy_high / total_energy if total_energy > 0 else 0, 3)
        }

    def _assess_signal_quality(self, signal: np.ndarray) -> str:
        signal_power = np.mean(signal ** 2)
        noise_estimate = np.std(np.diff(signal))
        snr_estimate = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10))

        if snr_estimate > 20:
            return "excellent"
        elif snr_estimate > 10:
            return "good"
        elif snr_estimate > 5:
            return "moderate"
        else:
            return "poor"

    def _generate_recommendations(
        self,
        mhr: float,
        variability: float,
        classification: Dict[str, Any]
    ) -> list:
        recommendations = []

        risk_level = classification.get("risk_level", "unknown")

        if risk_level == "moderado":
            recommendations.append("MHR fora da faixa esperada em repouso. Recomenda-se repetir a medição e observar sintomas.")
            recommendations.append("Se persistir alteração ou houver sintomas, procurar orientação médica.")
        elif risk_level == "baixo":
            recommendations.append("MHR dentro dos parâmetros esperados em repouso na gestação.")
            recommendations.append("Mantenha acompanhamento pré-natal conforme orientação.")
        else:
            recommendations.append("Não foi possível classificar a MHR com segurança.")

        if variability < 5:
            recommendations.append("Variabilidade baixa detectada. Pode indicar necessidade de avaliação.")

        return recommendations

    def analyze_realtime_stream(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        return self._process_signal(audio_chunk, sample_rate)
