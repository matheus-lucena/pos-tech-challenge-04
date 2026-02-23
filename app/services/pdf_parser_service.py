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
from services.comprehend_medical_service import ComprehendMedicalService
from services.textract_service import TextractService


class PDFParserService:
    def __init__(self):
        self.textract_service = TextractService()
        self.comprehend_service = ComprehendMedicalService()
    
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
        age_match = re.search(r'age[:\s]*(\d+)', text_lower, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r'(\d+)\s*years?', text_lower)
        if age_match:
            try:
                age = float(age_match.group(1))
                if PDF_AGE_MIN <= age <= PDF_AGE_MAX:
                    form_data["age"] = age
            except (ValueError, TypeError):
                pass
        
        bp_patterns = [
            r'blood\s*pressure[:\s]*(\d+)[/\sxX](\d+)',
            r'bp[:\s]*(\d+)[/\sxX](\d+)',
            r'(\d+)[/\sxX](\d+)\s*mmhg',
            r'(\d+)[/\sxX](\d+)\s*mm\s*hg'
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
        
        glucose_patterns = [
            r'glucose[:\s]*(\d+[.,]?\d*)',
            r'gluc[:\s]*(\d+[.,]?\d*)',
            r'bs[:\s]*(\d+[.,]?\d*)',
            r'(\d+[.,]?\d*)\s*mg/dl\s*(?:glucose)',
        ]
        
        for pattern in glucose_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    glucose_str = match.group(1).replace(",", ".")
                    glucose = float(glucose_str)
                    if PDF_GLUCOSE_MIN <= glucose <= PDF_GLUCOSE_MAX:
                        form_data["glucose"] = glucose
                        break
                except (ValueError, TypeError):
                    continue
        
        temp_patterns = [
            r'temperature[:\s]*(\d+[.,]?\d*)\s*[°]?[fcFC]',
            r'temp[:\s]*(\d+[.,]?\d*)\s*[°]?[fcFC]',
            r'(\d+[.,]?\d*)\s*[°]?[fcFC]\s*(?:temperature|temp)',
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    temp_str = match.group(1).replace(",", ".")
                    temp = float(temp_str)
                    if "f" in match.group(0).lower():
                        if PDF_TEMP_FAHRENHEIT_MIN <= temp <= PDF_TEMP_FAHRENHEIT_MAX:
                            form_data["temperature"] = temp
                    else:
                        if PDF_TEMP_CELSIUS_MIN <= temp <= PDF_TEMP_CELSIUS_MAX:
                            form_data["temperature"] = (temp * 9 / 5) + 32
                    if form_data["temperature"]:
                        break
                except (ValueError, TypeError):
                    continue
        
        hr_patterns = [
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
