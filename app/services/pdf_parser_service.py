import re
from typing import Dict, Any, Optional

from config.constants import (
    PDF_AGE_MIN,
    PDF_AGE_MAX,
    PDF_BP_SYSTOLIC_MIN,
    PDF_BP_SYSTOLIC_MAX,
    PDF_BP_DIASTOLIC_MIN,
    PDF_BP_DIASTOLIC_MAX,
    PDF_GLUCOSE_MIN,
    PDF_GLUCOSE_MAX,
    PDF_TEMP_FAHRENHEIT_MIN,
    PDF_TEMP_FAHRENHEIT_MAX,
    PDF_TEMP_CELSIUS_MIN,
    PDF_TEMP_CELSIUS_MAX,
    PDF_HR_MIN,
    PDF_HR_MAX,
)
from typing import Optional

from services.comprehend_medical_service import ComprehendMedicalService
from services.textract_service import TextractService


class PDFParserService:
    def __init__(
        self,
        textract_service: Optional[TextractService] = None,
        comprehend_service: Optional[ComprehendMedicalService] = None,
    ):
        self.textract_service = textract_service or TextractService()
        self.comprehend_service = comprehend_service or ComprehendMedicalService()
    
    def extract_medical_data_from_pdf(
        self, 
        pdf_path: str,
        is_s3_path: bool = False
    ) -> Dict[str, Any]:
        if is_s3_path:
            text = self.textract_service.extract_text_from_pdf_s3(pdf_path)
        else:
            text = self.textract_service.extract_text_from_pdf_local(pdf_path)
        
        if not text:
            return {
                "success": False,
                "error": "Could not extract text from PDF",
                "form_data": {}
            }
        
        try:
            entities = self.comprehend_service.detect_entities(text)
        except Exception as e:
            print(f"Error analyzing with Comprehend Medical: {str(e)}")
            entities = {}
        
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
        form_data = {
            "age": None,
            "systolic_bp": None,
            "diastolic_bp": None,
            "glucose": None,
            "temperature": None,
            "heart_rate": None
        }
        
        text_lower = text.lower()

        # ── Age ──────────────────────────────────────────────────────────────
        # Supports English ("35 years") and Portuguese ("(35 Anos)" / "35 anos")
        age_patterns = [
            r'age[:\s]*(\d+)',
            r'(\d+)\s*years?\b',
            r'\((\d+)\s*anos?\b',   # "(35 Anos)" — common in Brazilian reports
            r'\b(\d+)\s*anos?\b',   # "35 anos"
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, text_lower)
            if age_match:
                try:
                    age = float(age_match.group(1))
                    if PDF_AGE_MIN <= age <= PDF_AGE_MAX:
                        form_data["age"] = age
                        break
                except (ValueError, TypeError):
                    continue

        # ── Blood pressure ───────────────────────────────────────────────────
        bp_patterns = [
            r'press[aã]o\s+arterial[^:\n]*:\s*(\d+)\s*[xX/]\s*(\d+)',  # PT "Pressão Arterial: 124x81"
            r'blood\s*pressure[:\s]*(\d+)[/\sxX](\d+)',
            r'bp[:\s]*(\d+)[/\sxX](\d+)',
            r'(\d+)\s*[xX/]\s*(\d+)\s*mmhg',
            r'(\d+)\s*[xX/]\s*(\d+)\s*mm\s*hg',
        ]
        for pattern in bp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    systolic = float(match.group(1))
                    diastolic = float(match.group(2))
                    if (
                        PDF_BP_SYSTOLIC_MIN <= systolic <= PDF_BP_SYSTOLIC_MAX
                        and PDF_BP_DIASTOLIC_MIN <= diastolic <= PDF_BP_DIASTOLIC_MAX
                        and systolic > diastolic
                    ):
                        form_data["systolic_bp"] = systolic
                        form_data["diastolic_bp"] = diastolic
                        break
                except (ValueError, TypeError):
                    continue

        # ── Glucose ───────────────────────────────────────────────────────────
        # PDF reports often use mg/dL (e.g. "95 mg/dL"); the model expects mmol/L.
        # Values >30 are treated as mg/dL and converted (÷18.018).
        _MG_DL_THRESHOLD = 30.0
        _MG_DL_TO_MMOL = 18.018
        glucose_patterns = [
            r'glicemia[^:\n]*:\s*(\d+[.,]?\d*)',          # PT "Glicemia de Jejum: 95"
            r'glucose[^:\n]*:\s*(\d+[.,]?\d*)',
            r'gluc[:\s]*(\d+[.,]?\d*)',
            r'bs[:\s]*(\d+[.,]?\d*)',
            r'(\d+[.,]?\d*)\s*mg/dl',                     # standalone "95 mg/dL"
        ]
        for pattern in glucose_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    glucose = float(match.group(1).replace(",", "."))
                    # Auto-convert mg/dL → mmol/L when value is clearly in mg/dL range
                    if glucose > _MG_DL_THRESHOLD:
                        glucose = round(glucose / _MG_DL_TO_MMOL, 2)
                    if PDF_GLUCOSE_MIN <= glucose <= PDF_GLUCOSE_MAX:
                        form_data["glucose"] = glucose
                        break
                except (ValueError, TypeError):
                    continue

        # ── Temperature ───────────────────────────────────────────────────────
        # Supports Celsius (°C) — auto-converts to Fahrenheit — and direct °F values.
        # Matches Portuguese "Temperatura Axilar: 37.2 °C" as well as English forms.
        # Each tuple is (pattern, unit) where unit is "C", "F", or "auto".
        # "auto" = decide by value range (Celsius: 35-40.5 / Fahrenheit: 95-105).
        temp_patterns = [
            (r'temperatura[^:\n]*:\s*(\d+[.,]\d+)\s*graus?\s*[cC]', "C"),   # "37.5 graus C"
            (r'temperatura[^:\n]*:\s*(\d+[.,]\d+)\s*°\s*[cC]', "C"),        # "37.5 °C"
            (r'temperatura[^:\n]*:\s*(\d+[.,]\d+)\s*graus?\s*[fF]', "F"),   # "98.6 graus F"
            (r'temperatura[^:\n]*:\s*(\d+[.,]\d+)\s*°\s*[fF]', "F"),        # "98.6 °F"
            (r'temperatura[^:\n]*:\s*(\d+[.,]\d+)', "auto"),                 # decimal, no unit
            (r'temperatura[^:\n]*:\s*(\d+)', "auto"),                        # integer, no unit
            (r'temperature[^:\n]*:\s*(\d+[.,]?\d*)\s*[°]?\s*[fF]', "F"),
            (r'temperature[^:\n]*:\s*(\d+[.,]?\d*)\s*[°]?\s*[cC]', "C"),
            (r'(\d+[.,]?\d*)\s*°\s*[cC]\b', "C"),
            (r'(\d+[.,]?\d*)\s*°\s*[fF]\b', "F"),
        ]
        for pattern, unit in temp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    temp = float(match.group(1).replace(",", "."))
                    if unit == "F":
                        if PDF_TEMP_FAHRENHEIT_MIN <= temp <= PDF_TEMP_FAHRENHEIT_MAX:
                            form_data["temperature"] = temp
                            break
                    elif unit == "C":
                        if PDF_TEMP_CELSIUS_MIN <= temp <= PDF_TEMP_CELSIUS_MAX:
                            form_data["temperature"] = round((temp * 9 / 5) + 32, 1)
                            break
                    else:  # auto — decide by value range
                        if PDF_TEMP_CELSIUS_MIN <= temp <= PDF_TEMP_CELSIUS_MAX:
                            form_data["temperature"] = round((temp * 9 / 5) + 32, 1)
                            break
                        elif PDF_TEMP_FAHRENHEIT_MIN <= temp <= PDF_TEMP_FAHRENHEIT_MAX:
                            form_data["temperature"] = temp
                            break
                except (ValueError, TypeError):
                    continue

        # ── Heart rate ────────────────────────────────────────────────────────
        hr_patterns = [
            r'freq[uü][eê]ncia\s+card[ií]aca[^:\n]*:\s*(\d+)',  # PT "Frequência Cardíaca: 82"
            r'heart\s*rate[:\s]*(\d+)',
            r'hr[:\s]*(\d+)',
            r'(\d+)\s*bpm',
            r'(\d+)\s*beats',
        ]
        for pattern in hr_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    hr = float(match.group(1))
                    if PDF_HR_MIN <= hr <= PDF_HR_MAX:
                        form_data["heart_rate"] = hr
                        break
                except (ValueError, TypeError):
                    continue
        
        if entities and 'entities' in entities:
            for entity_type, entity_list in entities.get('entities', {}).items():
                for entity in entity_list:
                    text_entity = entity.get('text', '')
                    
                    if entity_type == "AGE" and not form_data["age"]:
                        try:
                            age_match = re.search(r"(\d+)", text_entity)
                            if age_match:
                                age = float(age_match.group(1))
                                if PDF_AGE_MIN <= age <= PDF_AGE_MAX:
                                    form_data["age"] = age
                        except (ValueError, TypeError):
                            pass
                    if entity_type == "TEST_VALUE" and not form_data["systolic_bp"]:
                        bp_match = re.search(r"(\d+)[/\sxX](\d+)", text_entity)
                        if bp_match:
                            try:
                                systolic = float(bp_match.group(1))
                                diastolic = float(bp_match.group(2))
                                if (
                                    PDF_BP_SYSTOLIC_MIN <= systolic <= PDF_BP_SYSTOLIC_MAX
                                    and PDF_BP_DIASTOLIC_MIN <= diastolic <= PDF_BP_DIASTOLIC_MAX
                                    and systolic > diastolic
                                ):
                                    form_data["systolic_bp"] = systolic
                                    form_data["diastolic_bp"] = diastolic
                            except (ValueError, TypeError):
                                pass
        
        return form_data
