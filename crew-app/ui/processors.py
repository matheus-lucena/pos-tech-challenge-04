"""Processadores para análise de dados."""

import json
import re
import traceback
from typing import Optional, Dict, Any, Tuple, Union
from services.s3_service import S3Service
from services.pdf_parser_service import PDFParserService
from agents.crew_orchestrator import iniciar_analise_multimodal
from config.llm_config import get_llm
from ui.formatters import formatar_resultado


class AnalysisProcessor:
    """Processador principal para análise de saúde materna."""
    
    def __init__(self):
        """Inicializa o processador com os serviços necessários."""
        self.s3_service = S3Service()
        self.llm = get_llm()
        self.pdf_parser = PDFParserService()
    
    def processar_analise(
        self,
        idade: Optional[float],
        pressao_sistolica: Optional[float],
        pressao_diastolica: Optional[float],
        glicemia: Optional[float],
        temperatura: Optional[float],
        frequencia_cardiaca: Optional[float],
        arquivo_audio: Optional[str],
        s3_audio: Optional[str],
        arquivo_audio_fetal: Optional[str] = None,
        s3_audio_fetal: Optional[str] = None
    ) -> str:
        """
        Processa a análise com os dados fornecidos.
        
        Args:
            idade: Idade da paciente
            pressao_sistolica: Pressão sistólica
            pressao_diastolica: Pressão diastólica
            glicemia: Nível de glicemia
            temperatura: Temperatura corporal
            frequencia_cardiaca: Frequência cardíaca
            arquivo_audio: Caminho do arquivo de áudio local (consulta/emocional)
            s3_audio: Caminho S3 do áudio (consulta/emocional)
            arquivo_audio_fetal: Caminho do arquivo de áudio fetal (PCG) local
            s3_audio_fetal: Caminho S3 do áudio fetal (PCG)
        
        Returns:
            String HTML com o resultado formatado
        """
        # Valida e prepara dados biométricos
        dados_biometria = self._preparar_dados_biometria(
            idade, pressao_sistolica, pressao_diastolica,
            glicemia, temperatura, frequencia_cardiaca
        )
        
        if isinstance(dados_biometria, str):
            return self._format_error(dados_biometria)
        
        # áudio de consulta/emocional
        audio_result = self._processar_audio(arquivo_audio, s3_audio)
        if isinstance(audio_result, str) and audio_result.startswith('<div'):
            return audio_result
        
        audio_path, status_msg = audio_result
        
        # (PCG)
        audio_fetal_result = self._processar_audio(arquivo_audio_fetal, s3_audio_fetal)
        if isinstance(audio_fetal_result, str) and audio_fetal_result.startswith('<div'):
            return audio_fetal_result
        
        audio_fetal_path, status_msg_fetal = audio_fetal_result
        
        combined_status_msg = status_msg
        if status_msg_fetal:
            combined_status_msg = (combined_status_msg or "") + status_msg_fetal
        
        # Valida se pelo menos um dado foi fornecido
        if not dados_biometria and not audio_path and not audio_fetal_path:
            return self._format_warning(
                "Por favor, forneça pelo menos dados biométricos, um arquivo de áudio de consulta ou um arquivo de áudio fetal (PCG)."
            )
        
        # Executa a análise
        try:
            resultado = iniciar_analise_multimodal(
                llm=self.llm,
                dados_biometria=dados_biometria,
                s3_audio=audio_path,
                s3_fetal_audio=audio_fetal_path
            )
            
            resultado_parsed = self._parse_resultado(resultado)
            resultado_formatado = formatar_resultado(resultado_parsed)
            
            if combined_status_msg:
                return combined_status_msg + resultado_formatado
            
            return resultado_formatado
        except Exception as e:
            return self._format_exception(e)
    
    def _preparar_dados_biometria(
        self,
        idade: Optional[float],
        pressao_sistolica: Optional[float],
        pressao_diastolica: Optional[float],
        glicemia: Optional[float],
        temperatura: Optional[float],
        frequencia_cardiaca: Optional[float]
    ) -> Union[Optional[Dict[str, Any]], str]:
        """
        Prepara e valida os dados biométricos.
        
        Returns:
            Dicionário com dados biométricos ou string de erro
        """
        if not all([idade, pressao_sistolica, pressao_diastolica, glicemia, temperatura, frequencia_cardiaca]):
            return None
        
        try:
            return {
                "Age": int(idade),
                "SystolicBP": int(pressao_sistolica),
                "DiastolicBP": int(pressao_diastolica),
                "BS": float(glicemia),
                "BodyTemp": float(temperatura),
                "HeartRate": int(frequencia_cardiaca)
            }
        except ValueError as e:
            return f"Valores inválidos nos dados biométricos. {str(e)}"
    
    def _processar_audio(
        self,
        arquivo_audio: Optional[str],
        s3_audio: Optional[str]
    ) -> Union[Tuple[Optional[str], str], str]:
        """
        Processa o áudio: upload ou validação de caminho S3.
        
        Returns:
            Tupla (caminho_audio, mensagem_status) ou string de erro HTML
        """
        audio_path = None
        status_msg = ""
        
        if arquivo_audio:
            status_msg = self._format_upload_status("Fazendo upload do áudio para S3...")
            
            s3_path = self.s3_service.upload_audio(arquivo_audio)
            if s3_path:
                audio_path = s3_path
                status_msg = self._format_success_status(f"Upload concluído: {s3_path}")
            else:
                return self._format_error(
                    "Falha ao fazer upload do áudio para S3. "
                    "Verifique as credenciais AWS e permissões."
                )
        
        elif s3_audio and s3_audio.strip():
            audio_path = s3_audio.strip()
            if not audio_path.startswith('s3://'):
                return self._format_error('O caminho S3 deve começar com "s3://"')
        
        return audio_path, status_msg
    
    def _parse_resultado(self, resultado: Any) -> Any:
        """
        Parseia o resultado do CrewAI para um formato utilizável.
        
        Args:
            resultado: Resultado bruto do CrewAI
        
        Returns:
            Resultado parseado (dict ou string)
        """
        if hasattr(resultado, 'raw'):
            resultado = resultado.raw
        elif hasattr(resultado, 'tasks_output'):
            if resultado.tasks_output:
                resultado = resultado.tasks_output[-1]
        elif hasattr(resultado, '__dict__'):
            resultado = resultado.__dict__
        
        if isinstance(resultado, str):
            resultado_str = resultado.strip()
            if resultado_str.startswith('{'):
                try:
                    resultado = json.loads(resultado_str)
                except:
                    json_match = re.search(r'\{.*\}', resultado_str, re.DOTALL)
                    if json_match:
                        try:
                            resultado = json.loads(json_match.group())
                        except:
                            pass
        
        return resultado
    
    def _format_error(self, message: str) -> str:
        """Formata uma mensagem de erro."""
        return (
            f'<div style="padding: 20px; background: #fff5f5; border-radius: 8px; color: #dc3545;">'
            f'<strong>❌ Erro:</strong> {message}</div>'
        )
    
    def _format_warning(self, message: str) -> str:
        """Formata uma mensagem de aviso."""
        return (
            f'<div style="padding: 20px; background: #fff3cd; border-radius: 8px; color: #856404;">'
            f'<strong>⚠️ Atenção:</strong> {message}</div>'
        )
    
    def _format_upload_status(self, message: str) -> str:
        """Formata uma mensagem de status de upload."""
        return (
            '<div style="padding: 15px; background: #e7f3ff; border-radius: 8px; '
            'margin-bottom: 15px; border-left: 4px solid #007bff;">'
            f'<p style="margin: 0; color: #004085;"><strong>📤</strong> {message}</p>'
            '</div>'
        )
    
    def _format_success_status(self, message: str) -> str:
        """Formata uma mensagem de sucesso."""
        return (
            '<div style="padding: 15px; background: #d4edda; border-radius: 8px; '
            'margin-bottom: 15px; border-left: 4px solid #28a745;">'
            f'<p style="margin: 0; color: #155724;"><strong>✅</strong> {message}</p>'
            '</div>'
        )
    
    def _format_exception(self, exception: Exception) -> str:
        """Formata uma exceção com traceback."""
        error_details = traceback.format_exc()
        return (
            '<div style="padding: 20px; background: #fff5f5; border-radius: 8px; color: #dc3545;">'
            f'<strong>❌ Erro ao processar análise:</strong><br>{str(exception)}<br><br>'
            f'<details><summary>Detalhes técnicos</summary><pre>{error_details}</pre></details>'
            '</div>'
        )


# Instância global do processador
_processor = AnalysisProcessor()


def processar_analise(*args) -> str:
    """Wrapper para processar análise usando a instância global."""
    return _processor.processar_analise(*args)


def processar_pdf_preenchimento(
    arquivo_pdf: Optional[str]
) -> Tuple[Optional[float], Optional[float], Optional[float], 
           Optional[float], Optional[float], Optional[float], str]:
    """
    Processa um PDF de exame médico e retorna os valores para pré-preenchimento.
    
    Args:
        arquivo_pdf: Caminho do arquivo PDF
    
    Returns:
        Tupla com (idade, pressao_sistolica, pressao_diastolica, 
                   glicemia, temperatura, frequencia_cardiaca, status_msg)
    """
    if not arquivo_pdf:
        return (
            None, None, None, None, None, None,
            '<div style="padding: 15px; background: #fff3cd; border-radius: 8px; '
            'margin-bottom: 15px;"><strong>⚠️</strong> Nenhum arquivo PDF fornecido.</div>'
        )
    
    try:
        # Extrai dados do PDF
        resultado = _processor.pdf_parser.extract_medical_data_from_pdf(
            pdf_path=arquivo_pdf,
            is_s3_path=False
        )
        
        if not resultado.get("success", False):
            error_msg = resultado.get("error", "Erro desconhecido ao processar PDF")
            return (
                None, None, None, None, None, None,
                f'<div style="padding: 15px; background: #fff5f5; border-radius: 8px; '
                f'margin-bottom: 15px; color: #dc3545;"><strong>❌ Erro:</strong> {error_msg}</div>'
            )
        
        form_data = resultado.get("form_data", {})
        
        campos_encontrados = []
        if form_data.get("idade"):
            campos_encontrados.append("Idade")
        if form_data.get("pressao_sistolica") and form_data.get("pressao_diastolica"):
            campos_encontrados.append("Pressão Arterial")
        if form_data.get("glicemia"):
            campos_encontrados.append("Glicemia")
        if form_data.get("temperatura"):
            campos_encontrados.append("Temperatura")
        if form_data.get("frequencia_cardiaca"):
            campos_encontrados.append("Frequência Cardíaca")
        
        if campos_encontrados:
            status_msg = (
                f'<div style="padding: 15px; background: #d4edda; border-radius: 8px; '
                f'margin-bottom: 15px; border-left: 4px solid #28a745;">'
                f'<p style="margin: 0; color: #155724;"><strong>✅ PDF processado com sucesso!</strong></p>'
                f'<p style="margin: 5px 0 0 0; color: #155724;">Campos encontrados: {", ".join(campos_encontrados)}</p>'
                f'</div>'
            )
        else:
            status_msg = (
                '<div style="padding: 15px; background: #fff3cd; border-radius: 8px; '
                'margin-bottom: 15px; border-left: 4px solid #ffc107;">'
                '<p style="margin: 0; color: #856404;"><strong>⚠️ PDF processado, mas nenhum dado biométrico foi encontrado.</strong></p>'
                '<p style="margin: 5px 0 0 0; color: #856404;">Por favor, preencha os campos manualmente.</p>'
                '</div>'
            )
        
        return (
            form_data.get("idade"),
            form_data.get("pressao_sistolica"),
            form_data.get("pressao_diastolica"),
            form_data.get("glicemia"),
            form_data.get("temperatura"),
            form_data.get("frequencia_cardiaca"),
            status_msg
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        return (
            None, None, None, None, None, None,
            f'<div style="padding: 15px; background: #fff5f5; border-radius: 8px; '
            f'margin-bottom: 15px; color: #dc3545;">'
            f'<strong>❌ Erro ao processar PDF:</strong> {str(e)}<br>'
            f'<details><summary>Detalhes técnicos</summary><pre>{error_details}</pre></details>'
            f'</div>'
        )

