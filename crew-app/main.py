import os
import boto3
import json
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from pydantic import BaseModel

class PlanoAcao(BaseModel):
    risco: str
    probabilidade: str
    recomendacoes: list[str]

# ============================================================================
# 1. CONFIGURAÇÃO DO LLM LOCAL (Ollama)
# ============================================================================
os.environ["OPENAI_API_KEY"] = "sk-ollama-local"

local_llm = LLM(
    model="ollama/qwen2.5:7b", 
    base_url="http://localhost:11434"
)

# ============================================================================
# 2. DEFINIÇÃO DA TOOL (FERRAMENTA Sagemaker)
# ============================================================================
@tool("MaternalRiskPredictor")
def predict_risk(data_json: str):
    """
    Envia sinais vitais para o SageMaker e retorna o risco.
    Input esperado (JSON): {"Age": 35, "SystolicBP": 140, "DiastolicBP": 90, "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 70}
    """
    if isinstance(data_json, dict):
        payload = data_json
    else:
        try:
            data_json = data_json.replace("```json", "").replace("```", "").strip()
            payload = json.loads(data_json)
        except:
            return "Erro: O input deve ser um JSON válido."

    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT", "sagemaker-xgboost-2026-01-22-21-24-26-641")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    try:
        client = boto3.client("sagemaker-runtime", region_name=region)
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        risk_status = "ALTO RISCO" if result.get("maternal_health_risk") == 1 else "BAIXO RISCO"
        
        prob_raw = result.get("risk_probability", "0")
        if isinstance(prob_raw, str) and '%' in prob_raw:
            prob_display = prob_raw
        else:
            prob_display = f"{float(prob_raw):.2%}"
        
        return f"Resultado da Predição: {risk_status} (Probabilidade: {prob_display})"
    except Exception as e:
        return f"Erro ao invocar SageMaker: {str(e)}"

# ============================================================================
# 3. DEFINIÇÃO DOS AGENTES
# ============================================================================
def create_health_crew(patient_data: dict):
    # Agente 1: allow_delegation=False para evitar loops
    analyst_agent = Agent(
        role='Analista de Dados Médicos',
        goal='Interpretar os sinais vitais e obter a classificação de risco do modelo no SageMaker.',
        backstory="""Você é um especialista em bioestatística. Sua única função é usar a ferramenta 
        MaternalRiskPredictor para diagnosticar a paciente. Responda sempre em Português.""",
        tools=[predict_risk],
        llm=local_llm,
        allow_delegation=False,
        max_iter=3,
        verbose=True
    )

    # Agente 2: Médico
    doctor_agent = Agent(
        role='Obstetra Especialista',
        goal='Fornecer recomendações médicas baseadas no nível de risco detectado pelo analista.',
        backstory="""Você é um médico obstetra sênior. Baseie suas recomendações EXCLUSIVAMENTE 
        no relatório do analista. Responda sempre em Português do Brasil.""",
        llm=local_llm,
        allow_delegation=False,
        max_iter=3,
        verbose=True
    )

    # ============================================================================
    # 4. DEFINIÇÃO DAS TAREFAS
    # ============================================================================
    task_analysis = Task(
        description=f"Analise os sinais vitais da paciente: {patient_data}. Use a ferramenta MaternalRiskPredictor.",
        expected_output="Um relatório técnico com Nível de Risco e Probabilidade.",
        agent=analyst_agent
    )

    task_recommendation = Task(
        description="Com base na análise técnica anterior, elabore um plano de ação médico simplificado.",
        expected_output="Resumo executivo em Português com: Risco, Preocupações e Recomendações Clínicas.",
        agent=doctor_agent,
        context=[task_analysis],
        output_json=PlanoAcao
    )

    # ============================================================================
    # 5. EXECUÇÃO
    # ============================================================================
    crew = Crew(
        agents=[analyst_agent, doctor_agent],
        tasks=[task_analysis, task_recommendation],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()

if __name__ == "__main__":
    exemplo_paciente = {
        "Age": 35, 
        "SystolicBP": 140, 
        "DiastolicBP": 90, 
        "BS": 13.0, 
        "BodyTemp": 98.0, 
        "HeartRate": 70
    }
    
    print("\n--- Iniciando Processamento da Crew ---")
    resultado = create_health_crew(exemplo_paciente)
    print("\n--- RESULTADO FINAL ---")
    print(resultado)