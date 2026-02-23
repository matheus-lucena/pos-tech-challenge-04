import os
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import boto3

load_dotenv()


class ComprehendMedicalService:
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client('comprehendmedical', region_name=self.region_name)
    
    def detect_entities(self, text: str) -> Dict[str, Any]:
        try:
            response = self.client.detect_entities(Text=text)
            
            entities_by_type = {}
            for entity in response.get('Entities', []):
                entity_type = entity.get('Type', 'UNKNOWN')
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                entities_by_type[entity_type].append({
                    'text': entity.get('Text', ''),
                    'category': entity.get('Category', ''),
                    'type': entity_type,
                    'score': entity.get('Score', 0),
                    'begin_offset': entity.get('BeginOffset', 0),
                    'end_offset': entity.get('EndOffset', 0),
                    'traits': entity.get('Traits', [])
                })
            
            return {
                'entities': entities_by_type,
                'total_entities': len(response.get('Entities', [])),
                'raw_response': response
            }
        except Exception as e:
            raise Exception(f"Comprehend Medical analysis error: {str(e)}")
    
    def detect_phi(self, text: str) -> Dict[str, Any]:
        try:
            response = self.client.detect_phi(Text=text)
            
            phi_by_type = {}
            for entity in response.get('Entities', []):
                entity_type = entity.get('Type', 'UNKNOWN')
                if entity_type not in phi_by_type:
                    phi_by_type[entity_type] = []
                
                phi_by_type[entity_type].append({
                    'text': entity.get('Text', ''),
                    'category': entity.get('Category', ''),
                    'type': entity_type,
                    'score': entity.get('Score', 0),
                    'begin_offset': entity.get('BeginOffset', 0),
                    'end_offset': entity.get('EndOffset', 0),
                    'traits': entity.get('Traits', [])
                })
            
            return {
                'phi': phi_by_type,
                'total_phi': len(response.get('Entities', [])),
                'raw_response': response
            }
        except Exception as e:
            raise Exception(f"PHI detection error: {str(e)}")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        try:
            entities_result = self.detect_entities(text)
            phi_result = self.detect_phi(text)
            
            return {
                'entities': entities_result['entities'],
                'total_entities': entities_result['total_entities'],
                'phi': phi_result['phi'],
                'total_phi': phi_result['total_phi'],
                'summary': {
                    'entity_types': list(entities_result['entities'].keys()),
                    'phi_types': list(phi_result['phi'].keys()),
                    'has_medical_info': entities_result['total_entities'] > 0,
                    'has_phi': phi_result['total_phi'] > 0
                }
            }
        except Exception as e:
            raise Exception(f"Complete analysis error: {str(e)}")
    
    def format_analysis_result(self, analysis: Dict[str, Any]) -> str:
        lines = []
        lines.append("=== COMPREHEND MEDICAL ANALYSIS ===\n")
        
        lines.append(f"📋 MEDICAL ENTITIES DETECTED: {analysis['total_entities']}\n")
        for entity_type, entities in analysis['entities'].items():
            lines.append(f"\n🔹 {entity_type}:")
            for entity in entities:
                lines.append(f"   - Text: '{entity['text']}'")
                lines.append(f"     Category: {entity['category']}")
                lines.append(f"     Confidence: {entity['score']:.2%}")
                if entity['traits']:
                    traits = [t.get('Name', '') for t in entity['traits']]
                    lines.append(f"     Traits: {', '.join(traits)}")
        
        lines.append(f"\n🔒 PROTECTED INFORMATION (PHI): {analysis['total_phi']}\n")
        if analysis['total_phi'] > 0:
            for phi_type, phi_list in analysis['phi'].items():
                lines.append(f"\n🔸 {phi_type}:")
                for phi in phi_list:
                    lines.append(f"   - Text: '{phi['text']}'")
                    lines.append(f"     Category: {phi['category']}")
                    lines.append(f"     Confidence: {phi['score']:.2%}")
        else:
            lines.append("   No protected information detected.")
        
        lines.append("\n📊 SUMMARY:")
        lines.append(f"   - Entity types: {', '.join(analysis['summary']['entity_types']) if analysis['summary']['entity_types'] else 'None'}")
        lines.append(f"   - PHI types: {', '.join(analysis['summary']['phi_types']) if analysis['summary']['phi_types'] else 'None'}")
        lines.append(f"   - Contains medical info: {'Yes' if analysis['summary']['has_medical_info'] else 'No'}")
        lines.append(f"   - Contains PHI: {'Yes' if analysis['summary']['has_phi'] else 'No'}")
        
        return "\n".join(lines)
