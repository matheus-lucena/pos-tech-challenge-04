import json
import re
import traceback
from typing import Optional, Dict, Any, Tuple, Union
from services.s3_service import S3Service
from services.pdf_parser_service import PDFParserService
from services.maternal_health_service import MaternalHealthService
from agents.crew_orchestrator import start_multimodal_analysis
from config.llm_config import get_llm
from ui.formatters import format_result


class AnalysisProcessor:
    def __init__(self):
        self.s3_service = S3Service()
        self._llm = None
        self.pdf_parser = PDFParserService()
    
    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    def process_analysis(
        self,
        age: Optional[float],
        systolic_bp: Optional[float],
        diastolic_bp: Optional[float],
        glucose: Optional[float],
        temperature: Optional[float],
        heart_rate: Optional[float],
        audio_file: Optional[str],
        maternal_audio_file: Optional[str] = None,
    ) -> str:
        biometric_data = self._prepare_biometric_data(
            age, systolic_bp, diastolic_bp,
            glucose, temperature, heart_rate
        )
        
        if isinstance(biometric_data, str):
            return self._format_error(biometric_data)
        
        audio_result = self._process_audio(audio_file)
        if isinstance(audio_result, str) and audio_result.startswith('<div'):
            return audio_result
        
        audio_path, status_msg = audio_result
        
        maternal_audio_result = self._process_audio(maternal_audio_file)
        if isinstance(maternal_audio_result, str) and maternal_audio_result.startswith('<div'):
            return maternal_audio_result
        
        maternal_audio_path, status_msg_maternal = maternal_audio_result
        
        combined_status_msg = status_msg
        if status_msg_maternal:
            combined_status_msg = (combined_status_msg or "") + status_msg_maternal
        
        if not biometric_data and not audio_path and not maternal_audio_path:
            return self._format_warning(
                "Please provide at least biometric data, a consultation audio file or a maternal audio file (PCG)."
            )
        
        try:
            result = start_multimodal_analysis(
                llm=self.llm,
                biometric_data=biometric_data,
                s3_audio=None,  # Removido do fluxo principal
                s3_maternal_audio=maternal_audio_path
            )
            
            parsed_result = self._parse_result(result)
            formatted_result = format_result(parsed_result)
            
            if combined_status_msg:
                return combined_status_msg + formatted_result
            
            return formatted_result
        except Exception as e:
            return self._format_exception(e)
    
    def _prepare_biometric_data(
        self,
        age: Optional[float],
        systolic_bp: Optional[float],
        diastolic_bp: Optional[float],
        glucose: Optional[float],
        temperature: Optional[float],
        heart_rate: Optional[float]
    ) -> Union[Optional[Dict[str, Any]], str]:
        if not all([age, systolic_bp, diastolic_bp, glucose, temperature, heart_rate]):
            return None
        
        try:
            return {
                "Age": int(age),
                "SystolicBP": int(systolic_bp),
                "DiastolicBP": int(diastolic_bp),
                "BS": float(glucose),
                "BodyTemp": float(temperature),
                "HeartRate": int(heart_rate)
            }
        except ValueError as e:
            return f"Invalid values in biometric data. {str(e)}"
    
    def _process_audio(
        self,
        audio_file: Optional[str],
    ) -> Union[Tuple[Optional[str], str], str]:
        audio_path = None
        status_msg = ""
        
        if audio_file:
            status_msg = self._format_upload_status("Uploading audio to S3...")
            
            s3_path = self.s3_service.upload_audio(audio_file)
            if s3_path:
                audio_path = s3_path
                status_msg = self._format_success_status(f"Upload completed: {s3_path}")
            else:
                return self._format_error(
                    "Failed to upload audio to S3. "
                    "Check AWS credentials and permissions."
                )
        
        return audio_path, status_msg
    
    def _parse_result(self, result: Any) -> Any:
        if hasattr(result, 'raw'):
            result = result.raw
        elif hasattr(result, 'tasks_output'):
            if result.tasks_output:
                result = result.tasks_output[-1]
        elif hasattr(result, '__dict__'):
            result = result.__dict__
        
        if isinstance(result, str):
            result_str = result.strip()
            if result_str.startswith('{'):
                try:
                    result = json.loads(result_str)
                except:
                    json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                        except:
                            pass
        
        return result
    
    def _format_error(self, message: str) -> str:
        return (
            f'<div style="padding: 20px; background: #fff5f5; border-radius: 8px; color: #dc3545;">'
            f'<strong>❌ Error:</strong> {message}</div>'
        )
    
    def _format_warning(self, message: str) -> str:
        return (
            f'<div style="padding: 20px; background: #fff3cd; border-radius: 8px; color: #856404;">'
            f'<strong>⚠️ Warning:</strong> {message}</div>'
        )
    
    def _format_upload_status(self, message: str) -> str:
        return (
            '<div style="padding: 15px; background: #e7f3ff; border-radius: 8px; '
            'margin-bottom: 15px; border-left: 4px solid #007bff;">'
            f'<p style="margin: 0; color: #004085;"><strong>📤</strong> {message}</p>'
            '</div>'
        )
    
    def _format_success_status(self, message: str) -> str:
        return (
            '<div style="padding: 15px; background: #d4edda; border-radius: 8px; '
            'margin-bottom: 15px; border-left: 4px solid #28a745;">'
            f'<p style="margin: 0; color: #155724;"><strong>✅</strong> {message}</p>'
            '</div>'
        )
    
    def _format_exception(self, exception: Exception) -> str:
        error_details = traceback.format_exc()
        return (
            '<div style="padding: 20px; background: #fff5f5; border-radius: 8px; color: #dc3545;">'
            f'<strong>❌ Error processing analysis:</strong><br>{str(exception)}<br><br>'
            f'<details><summary>Technical details</summary><pre>{error_details}</pre></details>'
            '</div>'
        )


_processor = AnalysisProcessor()
_maternal_service = MaternalHealthService()


def process_analysis(*args) -> str:
    return _processor.process_analysis(*args)


def process_maternal_beats(
    maternal_audio_file: Optional[str],
) -> Tuple[str, Optional[float]]:
    """
    Processa automaticamente o sinal materno (PCG) para estimar a frequência
    cardíaca materna e a quantidade de batimentos assim que o arquivo é enviado.
    """
    if not maternal_audio_file:
        return (
            _processor._format_warning(
                "Nenhum arquivo de áudio materno (PCG) foi fornecido."
            ),
            None,
        )

    try:
        result = _maternal_service.analyze_maternal_signal(maternal_audio_file)
        if result.get("status") == "error":
            return (
                _processor._format_error(
                    result.get("error", "Erro desconhecido ao analisar sinal materno (PCG).")
                ),
                None,
            )

        num_beats = result.get("num_beats_detected", 0)
        mhr = float(result.get("maternal_heart_rate", 0.0) or 0.0)
        classification = result.get("classification", {}) or {}
        risk_level = classification.get("risk_level", "desconhecido")
        status = classification.get("status", "indeterminado")
        description = classification.get("description", "")

        return (
            '<div style="padding: 15px; background: #e8f4ff; border-radius: 8px; '
            'margin-bottom: 15px; border-left: 4px solid #17a2b8;">'
            '<p style="margin: 0; color: #004085;">'
            '<strong>🤰 Análise rápida de sinal materno (PCG)</strong><br>'
            f'Batimentos detectados: <strong>{num_beats}</strong><br>'
            f'Frequência cardíaca materna estimada (MHR): <strong>{mhr:.1f} bpm</strong><br>'
            f'Classificação: <strong>{status}</strong> (risco {risk_level})<br>'
            f'{description}'
            '</p>'
            '</div>',
            mhr,
        )

    except Exception as e:
        print(f"Error processing maternal beats: {e}")
        print(traceback.format_exc())
        return _processor._format_exception(e), None

def process_pdf_fill(
    pdf_file: Optional[str]
) -> Tuple[Optional[float], Optional[float], Optional[float], 
           Optional[float], Optional[float], Optional[float], str]:
    if not pdf_file:
        return (
            None, None, None, None, None, None,
            '<div style="padding: 15px; background: #fff3cd; border-radius: 8px; '
            'margin-bottom: 15px;"><strong>⚠️</strong> No PDF file provided.</div>'
        )
    
    try:
        result = _processor.pdf_parser.extract_medical_data_from_pdf(
            pdf_path=pdf_file,
            is_s3_path=False
        )
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error processing PDF")
            return (
                None, None, None, None, None, None,
                f'<div style="padding: 15px; background: #fff5f5; border-radius: 8px; '
                f'margin-bottom: 15px; color: #dc3545;"><strong>❌ Error:</strong> {error_msg}</div>'
            )
        
        form_data = result.get("form_data", {})
        
        found_fields = []
        if form_data.get("age"):
            found_fields.append("Age")
        if form_data.get("systolic_bp") and form_data.get("diastolic_bp"):
            found_fields.append("Blood Pressure")
        if form_data.get("glucose"):
            found_fields.append("Glucose")
        if form_data.get("temperature"):
            found_fields.append("Temperature")
        if form_data.get("heart_rate"):
            found_fields.append("Heart Rate")
        
        if found_fields:
            status_msg = (
                f'<div style="padding: 15px; background: #d4edda; border-radius: 8px; '
                f'margin-bottom: 15px; border-left: 4px solid #28a745;">'
                f'<p style="margin: 0; color: #155724;"><strong>✅ PDF processed successfully!</strong></p>'
                f'<p style="margin: 5px 0 0 0; color: #155724;">Fields found: {", ".join(found_fields)}</p>'
                f'</div>'
            )
        else:
            status_msg = (
                '<div style="padding: 15px; background: #fff3cd; border-radius: 8px; '
                'margin-bottom: 15px; border-left: 4px solid #ffc107;">'
                '<p style="margin: 0; color: #856404;"><strong>⚠️ PDF processed, but no biometric data was found.</strong></p>'
                '<p style="margin: 5px 0 0 0; color: #856404;">Please fill in the fields manually.</p>'
                '</div>'
            )
        
        return (
            form_data.get("age"),
            form_data.get("systolic_bp"),
            form_data.get("diastolic_bp"),
            form_data.get("glucose"),
            form_data.get("temperature"),
            form_data.get("heart_rate"),
            status_msg
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        return (
            None, None, None, None, None, None,
            f'<div style="padding: 15px; background: #fff5f5; border-radius: 8px; '
            f'margin-bottom: 15px; color: #dc3545;">'
            f'<strong>❌ Error processing PDF:</strong> {str(e)}<br>'
            f'<details><summary>Technical details</summary><pre>{error_details}</pre></details>'
            f'</div>'
        )
