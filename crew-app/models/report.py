from typing import List
from pydantic import BaseModel, Field


class HealthReport(BaseModel):
    biometric_analysis: str = Field(..., description="Análise biométrica do modelo SageMaker. DEVE estar em PORTUGUÊS (português brasileiro).")
    emotional_analysis: str = Field(default="Não aplicável - análise emocional é apenas para áudio de consulta separado", description="Análise emocional baseada em áudio/transcrição. No fluxo principal, deve ser 'Não aplicável'.")
    maternal_analysis: str = Field(default="Não fornecido", description="Análise de sinal materno que DEVE INTEGRAR o resultado do modelo SageMaker (risco alto ou baixo) com a análise do sinal PCG. Se o SageMaker indicar ALTO RISCO, a análise materna DEVE refletir isso, mesmo que o PCG pareça normal. DEVE usar a frequência cardíaca dos dados biométricos fornecidos (HeartRate). DEVE estar em PORTUGUÊS (português brasileiro).")
    final_risk: str = Field(..., description="Classificação final de risco. DEVE corresponder ao resultado do modelo SageMaker: se SageMaker retornou HIGH RISK, usar 'ALTO RISCO'; se LOW RISK, usar 'BAIXO RISCO' (em português).")
    recommendations: List[str] = Field(..., description="Lista de RECOMENDAÇÕES (orientações, sugestões) em PORTUGUÊS. NÃO incluir procedimentos médicos específicos como 'administrar medicamento X' ou 'realizar cirurgia'. Apenas recomendações gerais como 'monitoramento contínuo recomendado', 'avaliação médica imediata recomendada', etc.")

