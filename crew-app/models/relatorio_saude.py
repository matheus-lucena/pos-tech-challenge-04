"""Modelo de dados para o relatório de saúde materna."""

from typing import List
from pydantic import BaseModel, Field


class RelatorioSaude(BaseModel):
    """Esquema de saída estruturada para o laudo médico."""
    
    analise_biometrica: str = Field(
        ..., 
        description="Resultado do modelo SageMaker"
    )
    analise_emocional: str = Field(
        ..., 
        description="Análise do áudio/transcrição"
    )
    analise_fetal: str = Field(
        default="Não fornecida",
        description="Análise de sinais fetais (FHR, variabilidade, classificação)"
    )
    risco_final: str = Field(
        ..., 
        description="Classificação final de risco"
    )
    recomendacoes: List[str] = Field(
        ..., 
        description="Lista de ações sugeridas"
    )

