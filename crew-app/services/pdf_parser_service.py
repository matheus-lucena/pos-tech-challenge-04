"""Serviço para parsear dados médicos de texto extraído de PDFs."""

import re
from typing import Dict, Any, Optional
from services.textract_service import TextractService
from services.comprehend_medical_service import ComprehendMedicalService


class PDFParserService:
    """Serviço para extrair e parsear dados médicos de PDFs de exames."""
    
    def __init__(self):
        """Inicializa o serviço com Textract e Comprehend Medical."""
        self.textract_service = TextractService()
        self.comprehend_service = ComprehendMedicalService()
    
    def extract_medical_data_from_pdf(
        self, 
        pdf_path: str,
        is_s3_path: bool = False
    ) -> Dict[str, Any]:
        """
        Extrai dados médicos de um PDF de exame.
        
        Args:
            pdf_path: Caminho do PDF (local ou S3)
            is_s3_path: Se True, pdf_path é um caminho S3
        
        Returns:
            Dicionário com dados extraídos e campos do formulário
        """
        # Extrai texto do PDF
        if is_s3_path:
            text = self.textract_service.extract_text_from_pdf_s3(pdf_path)
        else:
            text = self.textract_service.extract_text_from_pdf_local(pdf_path)
        
        if not text:
            return {
                "success": False,
                "error": "Não foi possível extrair texto do PDF",
                "form_data": {}
            }
        
        # Analisa o texto com Comprehend Medical para extrair entidades médicas
        try:
            entities = self.comprehend_service.detect_entities(text)
        except Exception as e:
            print(f"Erro ao analisar com Comprehend Medical: {str(e)}")
            entities = {}
        
        # Parseia dados específicos do formulário
        form_data = self._parse_form_fields(text, entities)
        
        return {
            "success": True,
            "extracted_text": text,
            "entities": entities,
            "form_data": form_data
        }
    
    def _parse_form_fields(
        self, 
        text: str, 
        entities: Dict[str, Any]
    ) -> Dict[str, Optional[float]]:
        """
        Parseia campos específicos do formulário a partir do texto.
        
        Args:
            text: Texto extraído do PDF
            entities: Entidades detectadas pelo Comprehend Medical
        
        Returns:
            Dicionário com valores para preencher o formulário
        """
        form_data = {
            "idade": None,
            "pressao_sistolica": None,
            "pressao_diastolica": None,
            "glicemia": None,
            "temperatura": None,
            "frequencia_cardiaca": None
        }
        
        # Normaliza o texto para busca
        text_lower = text.lower()
        text_upper = text.upper()
        
        # Extrai idade
        idade_match = re.search(r'idade[:\s]*(\d+)', text_lower, re.IGNORECASE)
        if not idade_match:
            idade_match = re.search(r'(\d+)\s*anos?', text_lower)
        if idade_match:
            try:
                form_data["idade"] = float(idade_match.group(1))
            except:
                pass
        
        # Extrai pressão arterial (sistólica e diastólica)
        # Padrões: "140/90", "PA: 140/90", "Pressão: 140x90", etc.
        pa_patterns = [
            r'press[ãa]o[:\s]*(\d+)[/\sxX](\d+)',
            r'pa[:\s]*(\d+)[/\sxX](\d+)',
            r'(\d+)[/\sxX](\d+)\s*mmhg',
            r'(\d+)[/\sxX](\d+)\s*mm\s*hg'
        ]
        
        for pattern in pa_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    sistolica = float(match.group(1))
                    diastolica = float(match.group(2))
                    # Validação básica: sistólica deve ser maior que diastólica
                    if 50 <= sistolica <= 250 and 30 <= diastolica <= 180 and sistolica > diastolica:
                        form_data["pressao_sistolica"] = sistolica
                        form_data["pressao_diastolica"] = diastolica
                        break
                except:
                    continue
        
        # Extrai glicemia
        glicemia_patterns = [
            r'glicemia[:\s]*(\d+[.,]?\d*)',
            r'glicose[:\s]*(\d+[.,]?\d*)',
            r'glic[:\s]*(\d+[.,]?\d*)',
            r'bs[:\s]*(\d+[.,]?\d*)',
            r'(\d+[.,]?\d*)\s*mg/dl\s*(?:glicemia|glicose)',
        ]
        
        for pattern in glicemia_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    glicemia_str = match.group(1).replace(',', '.')
                    glicemia = float(glicemia_str)
                    if 3.0 <= glicemia <= 30.0:
                        form_data["glicemia"] = glicemia
                        break
                except:
                    continue
        
        # Extrai temperatura
        temp_patterns = [
            r'temperatura[:\s]*(\d+[.,]?\d*)\s*[°]?[fcFC]',
            r'temp[:\s]*(\d+[.,]?\d*)\s*[°]?[fcFC]',
            r'(\d+[.,]?\d*)\s*[°]?[fcFC]\s*(?:temperatura|temp)',
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    temp_str = match.group(1).replace(',', '.')
                    temp = float(temp_str)
                    # Detecta se é Fahrenheit ou Celsius
                    if 'f' in match.group(0).lower():
                        # Já está em Fahrenheit
                        if 95.0 <= temp <= 105.0:
                            form_data["temperatura"] = temp
                    else:
                        # Assume Celsius, converte para Fahrenheit
                        if 35.0 <= temp <= 40.5:
                            form_data["temperatura"] = (temp * 9/5) + 32
                    if form_data["temperatura"]:
                        break
                except:
                    continue
        
        # Extrai frequência cardíaca
        fc_patterns = [
            r'frequ[êe]ncia\s*cardiaca[:\s]*(\d+)',
            r'fc[:\s]*(\d+)',
            r'freq[:\s]*(\d+)',
            r'(\d+)\s*bpm',
            r'(\d+)\s*batimentos',
        ]
        
        for pattern in fc_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    fc = float(match.group(1))
                    if 40 <= fc <= 200:
                        form_data["frequencia_cardiaca"] = fc
                        break
                except:
                    continue
        
        # Tenta usar entidades do Comprehend Medical como fallback
        if entities and 'entities' in entities:
            # O formato retornado pelo Comprehend Medical é um dicionário por tipo
            # Exemplo: {'AGE': [{'text': '35', ...}], ...}
            for entity_type, entity_list in entities.get('entities', {}).items():
                for entity in entity_list:
                    text_entity = entity.get('text', '')
                    
                    # Idade
                    if entity_type == 'AGE' and not form_data["idade"]:
                        try:
                            age_match = re.search(r'(\d+)', text_entity)
                            if age_match:
                                age = float(age_match.group(1))
                                if 15 <= age <= 50:
                                    form_data["idade"] = age
                        except:
                            pass
                    
                    # Pressão arterial - pode estar em TEST_VALUE ou outros tipos
                    if entity_type == 'TEST_VALUE' and not form_data["pressao_sistolica"]:
                        # Comprehend Medical pode detectar valores de teste
                        # Tenta extrair pressão do texto
                        pa_match = re.search(r'(\d+)[/\sxX](\d+)', text_entity)
                        if pa_match:
                            try:
                                sistolica = float(pa_match.group(1))
                                diastolica = float(pa_match.group(2))
                                if 50 <= sistolica <= 250 and 30 <= diastolica <= 180 and sistolica > diastolica:
                                    form_data["pressao_sistolica"] = sistolica
                                    form_data["pressao_diastolica"] = diastolica
                            except:
                                pass
        
        return form_data

