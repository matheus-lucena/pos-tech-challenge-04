"""Serviço para análise de sinais fetais (PCG - Phonocardiogram)."""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FetalAnalysisService:
    """Serviço para análise de sinais de coração fetal em tempo real."""
    
    # Faixas normais de frequência cardíaca fetal (bpm)
    FHR_NORMAL_MIN = 110
    FHR_NORMAL_MAX = 160
    FHR_TACHYCARDIA = 160  # Acima disso é taquicardia
    FHR_BRADYCARDIA = 110  # Abaixo disso é bradicardia
    
    def __init__(self):
        """Inicializa o serviço de análise fetal."""
        self.sample_rate = 16000  # Taxa de amostragem padrão do SUFHSDB
    
    def analyze_fetal_signal(
        self,
        audio_path: str,
        sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analisa um sinal de áudio de PCG fetal.
        
        Args:
            audio_path: Caminho para o arquivo de áudio (local ou S3)
            sample_rate: Taxa de amostragem (opcional, usa padrão se não fornecido)
        
        Returns:
            Dicionário com resultados da análise
        """
        try:
            # Carrega o áudio
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # Tenta carregar o áudio
            try:
                y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            except Exception as e:
                # Se falhar, tenta com taxa de amostragem diferente
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                # Reamostra se necessário
                if sr != sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                    sr = sample_rate
            
            # Processa o sinal
            return self._process_signal(y, sr)
            
        except Exception as e:
            return {
                "error": f"Erro ao processar sinal: {str(e)}",
                "status": "error"
            }
    
    def _process_signal(self, signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Processa o sinal de áudio para extrair métricas fetais.
        
        Args:
            signal: Array numpy com o sinal de áudio
            sample_rate: Taxa de amostragem
        
        Returns:
            Dicionário com métricas extraídas
        """
        # Filtro passa-banda para frequências cardíacas fetais (20-1000 Hz)
        # Mas focamos em 20-500 Hz para melhor detecção
        sos = scipy.signal.butter(4, [20, 500], btype='band', fs=sample_rate, output='sos')
        filtered_signal = scipy.signal.sosfilt(sos, signal)
        
        # Remove DC offset
        filtered_signal = filtered_signal - np.mean(filtered_signal)
        
        # Normaliza
        if np.max(np.abs(filtered_signal)) > 0:
            filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
        
        # Calcula frequência cardíaca fetal (FHR)
        fhr, fhr_confidence = self._estimate_fhr(filtered_signal, sample_rate)
        
        # Detecta batimentos cardíacos
        beats = self._detect_heartbeats(filtered_signal, sample_rate)
        
        # Calcula variabilidade da frequência cardíaca
        variability = self._calculate_variability(beats, sample_rate)
        
        # Classifica o estado
        classification = self._classify_fhr(fhr, variability)
        
        # Extrai características espectrais
        spectral_features = self._extract_spectral_features(filtered_signal, sample_rate)
        
        return {
            "fetal_heart_rate": round(fhr, 2),
            "fhr_confidence": round(fhr_confidence, 2),
            "variability": round(variability, 2),
            "classification": classification,
            "num_beats_detected": len(beats),
            "signal_quality": self._assess_signal_quality(filtered_signal),
            "spectral_features": spectral_features,
            "status": "success",
            "recommendations": self._generate_recommendations(fhr, variability, classification)
        }
    
    def _estimate_fhr(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> Tuple[float, float]:
        """
        Estima a frequência cardíaca fetal (FHR) em bpm.
        
        Args:
            signal: Sinal filtrado
            sample_rate: Taxa de amostragem
        
        Returns:
            Tupla (fhr, confidence)
        """
        # Usa autocorrelação para encontrar periodicidade
        # FHR fetal está tipicamente entre 110-160 bpm (1.83-2.67 Hz)
        min_period = int(sample_rate * 60 / self.FHR_TACHYCARDIA)  # ~6 amostras a 16kHz
        max_period = int(sample_rate * 60 / self.FHR_BRADYCARDIA)  # ~8.7 amostras a 16kHz
        
        # Calcula autocorrelação
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Procura pico na faixa de FHR
        search_range = autocorr[min_period:max_period]
        if len(search_range) == 0:
            return 0.0, 0.0
        
        peak_idx = np.argmax(search_range) + min_period
        
        # Calcula FHR
        period_samples = peak_idx
        fhr = (sample_rate * 60) / period_samples if period_samples > 0 else 0
        
        # Calcula confiança baseada na altura do pico
        peak_value = autocorr[peak_idx]
        max_value = np.max(autocorr)
        confidence = min(peak_value / max_value if max_value > 0 else 0, 1.0)
        
        # Valida se está na faixa esperada
        if fhr < 60 or fhr > 200:
            # Tenta método alternativo usando FFT
            fhr, confidence = self._estimate_fhr_fft(signal, sample_rate)
        
        return fhr, confidence
    
    def _estimate_fhr_fft(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> Tuple[float, float]:
        """Estima FHR usando análise espectral (FFT)."""
        # Calcula FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Converte para bpm
        freqs_bpm = freqs * 60
        
        # Procura pico na faixa de FHR (110-160 bpm)
        mask = (freqs_bpm >= self.FHR_BRADYCARDIA) & (freqs_bpm <= self.FHR_TACHYCARDIA)
        if not np.any(mask):
            return 0.0, 0.0
        
        search_freqs = freqs_bpm[mask]
        search_magnitude = magnitude[mask]
        
        peak_idx = np.argmax(search_magnitude)
        fhr = search_freqs[peak_idx]
        
        # Confiança baseada na altura relativa do pico
        max_magnitude = np.max(magnitude)
        confidence = search_magnitude[peak_idx] / max_magnitude if max_magnitude > 0 else 0
        
        return fhr, confidence
    
    def _detect_heartbeats(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Detecta batimentos cardíacos no sinal.
        
        Args:
            signal: Sinal filtrado
            sample_rate: Taxa de amostragem
        
        Returns:
            Array com índices dos batimentos detectados
        """
        sos = scipy.signal.butter(4, [50, 300], btype='band', fs=sample_rate, output='sos')
        filtered = scipy.signal.sosfilt(sos, signal)
        
        min_distance = int(sample_rate * 60 / self.FHR_TACHYCARDIA)
        
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
        """
        Calcula a variabilidade da frequência cardíaca.
        
        Args:
            beats: Array com índices dos batimentos
            sample_rate: Taxa de amostragem
        
        Returns:
            Variabilidade em bpm
        """
        if len(beats) < 2:
            return 0.0
        
        rr_intervals = np.diff(beats) / sample_rate  # em segundos
        rr_intervals_bpm = 60 / rr_intervals  # converte para bpm
        
        variability = np.std(rr_intervals_bpm) if len(rr_intervals_bpm) > 1 else 0.0
        
        return variability
    
    def _classify_fhr(
        self,
        fhr: float,
        variability: float
    ) -> Dict[str, Any]:
        """
        Classifica o estado da frequência cardíaca fetal.
        
        Args:
            fhr: Frequência cardíaca fetal em bpm
            variability: Variabilidade em bpm
        
        Returns:
            Dicionário com classificação
        """
        if fhr == 0:
            return {
                "status": "indeterminado",
                "risk_level": "unknown",
                "description": "Não foi possível determinar a FHR"
            }
        
        # Classifica FHR
        if fhr < self.FHR_BRADYCARDIA:
            fhr_status = "bradicardia"
            risk_level = "alto"
            description = f"FHR abaixo do normal ({fhr:.1f} bpm). Requer atenção médica imediata."
        elif fhr > self.FHR_TACHYCARDIA:
            fhr_status = "taquicardia"
            risk_level = "moderado"
            description = f"FHR acima do normal ({fhr:.1f} bpm). Pode indicar estresse fetal."
        else:
            fhr_status = "normal"
            risk_level = "baixo"
            description = f"FHR dentro da faixa normal ({fhr:.1f} bpm)."
        
        # Avalia variabilidade
        if variability < 5:
            variability_status = "baixa"
            if risk_level == "baixo":
                risk_level = "moderado"
            description += " Variabilidade baixa detectada."
        elif variability > 25:
            variability_status = "alta"
            description += " Variabilidade alta (pode ser normal em fetos ativos)."
        else:
            variability_status = "normal"
        
        return {
            "status": fhr_status,
            "variability_status": variability_status,
            "risk_level": risk_level,
            "description": description
        }
    
    def _extract_spectral_features(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Extrai características espectrais do sinal.
        
        Args:
            signal: Sinal filtrado
            sample_rate: Taxa de amostragem
        
        Returns:
            Dicionário com características espectrais
        """
        # Calcula espectro de potência
        freqs, psd = scipy.signal.welch(signal, sample_rate, nperseg=min(2048, len(signal)))
        
        # Frequência central (centroid)
        centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        
        # Largura de banda espectral
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
        """
        Avalia a qualidade do sinal.
        
        Args:
            signal: Sinal filtrado
        
        Returns:
            String descrevendo a qualidade
        """
        # Calcula SNR aproximado
        signal_power = np.mean(signal ** 2)
        noise_estimate = np.std(np.diff(signal))
        snr_estimate = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10))
        
        if snr_estimate > 20:
            return "excelente"
        elif snr_estimate > 10:
            return "boa"
        elif snr_estimate > 5:
            return "moderada"
        else:
            return "ruim"
    
    def _generate_recommendations(
        self,
        fhr: float,
        variability: float,
        classification: Dict[str, Any]
    ) -> list:
        """
        Gera recomendações baseadas na análise.
        
        Args:
            fhr: Frequência cardíaca fetal
            variability: Variabilidade
            classification: Classificação do estado
        
        Returns:
            Lista de recomendações
        """
        recommendations = []
        
        risk_level = classification.get("risk_level", "unknown")
        status = classification.get("status", "indeterminado")
        
        if risk_level == "alto":
            recommendations.append("⚠️ ATENÇÃO: FHR anormal detectada. Recomenda-se avaliação médica imediata.")
            recommendations.append("Considere monitoramento contínuo e avaliação por obstetra.")
        elif risk_level == "moderado":
            recommendations.append("FHR fora da faixa normal. Recomenda-se monitoramento adicional.")
            recommendations.append("Considere repetir a medição em breve.")
        else:
            recommendations.append("FHR dentro dos parâmetros normais.")
            recommendations.append("Continue o monitoramento regular conforme orientação médica.")
        
        if variability < 5:
            recommendations.append("Variabilidade baixa detectada. Pode indicar necessidade de avaliação.")
        
        return recommendations
    
    def analyze_realtime_stream(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Analisa um chunk de áudio em tempo real.
        
        Args:
            audio_chunk: Array numpy com chunk de áudio
            sample_rate: Taxa de amostragem
        
        Returns:
            Dicionário com resultados da análise
        """
        return self._process_signal(audio_chunk, sample_rate)

