"""Serviço para interação com AWS Comprehend Medical."""

import os
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import boto3

# Garante que as variáveis de ambiente estão carregadas
load_dotenv()


class ComprehendMedicalService:
    """Serviço para análise de texto médico usando AWS Comprehend Medical."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Inicializa o serviço Comprehend Medical.
        
        Args:
            region_name: Região AWS (padrão: us-east-1)
        """
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client('comprehendmedical', region_name=self.region_name)
    
    def detect_entities(self, text: str) -> Dict[str, Any]:
        """
        Detecta entidades médicas no texto.
        
        Args:
            text: Texto a ser analisado
        
        Returns:
            Dicionário com entidades detectadas e informações relevantes
        """
        try:
            response = self.client.detect_entities(Text=text)
            
            # Organiza as entidades por tipo
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
            raise Exception(f"Erro na análise Comprehend Medical: {str(e)}")
    
    def detect_phi(self, text: str) -> Dict[str, Any]:
        """
        Detecta informações de saúde protegidas (PHI - Protected Health Information).
        
        Args:
            text: Texto a ser analisado
        
        Returns:
            Dicionário com informações PHI detectadas
        """
        try:
            response = self.client.detect_phi(Text=text)
            
            # Organiza as informações PHI por tipo
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
            raise Exception(f"Erro na detecção de PHI: {str(e)}")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Realiza análise completa do texto (entidades médicas + PHI).
        
        Args:
            text: Texto a ser analisado
        
        Returns:
            Dicionário com análise completa incluindo entidades e PHI
        """
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
            raise Exception(f"Erro na análise completa: {str(e)}")
    
    def format_analysis_result(self, analysis: Dict[str, Any]) -> str:
        """
        Formata o resultado da análise em uma string legível.
        
        Args:
            analysis: Resultado da análise do Comprehend Medical
        
        Returns:
            String formatada com os resultados
        """
        lines = []
        lines.append("=== ANÁLISE COMPREHEND MEDICAL ===\n")
        
        # Entidades médicas
        lines.append(f"📋 ENTIDADES MÉDICAS DETECTADAS: {analysis['total_entities']}\n")
        for entity_type, entities in analysis['entities'].items():
            lines.append(f"\n🔹 {entity_type}:")
            for entity in entities:
                lines.append(f"   - Texto: '{entity['text']}'")
                lines.append(f"     Categoria: {entity['category']}")
                lines.append(f"     Confiança: {entity['score']:.2%}")
                if entity['traits']:
                    traits = [t.get('Name', '') for t in entity['traits']]
                    lines.append(f"     Traços: {', '.join(traits)}")
        
        # PHI
        lines.append(f"\n🔒 INFORMAÇÕES PROTEGIDAS (PHI): {analysis['total_phi']}\n")
        if analysis['total_phi'] > 0:
            for phi_type, phi_list in analysis['phi'].items():
                lines.append(f"\n🔸 {phi_type}:")
                for phi in phi_list:
                    lines.append(f"   - Texto: '{phi['text']}'")
                    lines.append(f"     Categoria: {phi['category']}")
                    lines.append(f"     Confiança: {phi['score']:.2%}")
        else:
            lines.append("   Nenhuma informação protegida detectada.")
        
        # Resumo
        lines.append("\n📊 RESUMO:")
        lines.append(f"   - Tipos de entidades: {', '.join(analysis['summary']['entity_types']) if analysis['summary']['entity_types'] else 'Nenhuma'}")
        lines.append(f"   - Tipos de PHI: {', '.join(analysis['summary']['phi_types']) if analysis['summary']['phi_types'] else 'Nenhuma'}")
        lines.append(f"   - Contém informações médicas: {'Sim' if analysis['summary']['has_medical_info'] else 'Não'}")
        lines.append(f"   - Contém PHI: {'Sim' if analysis['summary']['has_phi'] else 'Não'}")
        
        return "\n".join(lines)

