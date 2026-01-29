import os
import json
import time
import boto3
from typing import List, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# ============================================================================
# 1. CONFIGURAÇÃO DE AMBIENTE E LLM
# ============================================================================
os.environ["OPENAI_API_KEY"] = "sk-ollama-local"

local_llm = LLM(
    model="ollama/qwen2.5:7b", 
    base_url="http://localhost:11434"
)

# Esquema de saída estruturada para o laudo médico
class RelatorioSaude(BaseModel):
    analise_biometrica: str = Field(..., description="Resultado do modelo SageMaker")
    analise_emocional: str = Field(..., description="Análise do áudio/transcrição")
    risco_final: str = Field(..., description="Classificação final de risco")
    recomendacoes: List[str] = Field(..., description="Lista de ações sugeridas")

# ============================================================================
# 2. FERRAMENTAS (TOOLS) - AWS INTEGRATION
# ============================================================================
class HealthTools:
    
    @tool("MaternalRiskPredictor")
    def predict_risk(data_json: str):
        """Analisa sinais vitais via SageMaker. Espera JSON com biometria."""
        endpoint = os.getenv("SAGEMAKER_ENDPOINT", "sagemaker-xgboost-2026-01-22-21-24-26-641")
        client = boto3.client("sagemaker-runtime", region_name="us-east-1")
        
        try:
            payload = json.loads(data_json.replace("'", '"')) if isinstance(data_json, str) else data_json
            response = client.invoke_endpoint(
                EndpointName=endpoint,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            res = json.loads(response['Body'].read().decode())
            status = "ALTO RISCO" if res.get("maternal_health_risk") == 1 else "BAIXO RISCO"
            return f"Status: {status} | Confiança: {res.get('risk_probability', 0)}"
        except Exception as e:
            return f"Erro na predição: {str(e)}"

    @tool("AudioTranscriber")
    def transcribe_consultation(s3_path: str):
        """Inicia e recupera transcrição do Amazon Transcribe. s3_path ex: 's3://bucket/audio.mp3'"""
        import urllib.request
        
        transcribe = boto3.client('transcribe', region_name="us-east-1")
        s3_client = boto3.client('s3', region_name="us-east-1")
        job_name = f"job_{int(time.time())}"
        
        if not s3_path.startswith('s3://'):
            return f"Erro: S3 path deve começar com 's3://'. Recebido: {s3_path}"
        
        parts = s3_path.replace('s3://', '').split('/', 1)
        if len(parts) != 2:
            return f"Erro: Formato de S3 path inválido. Use: s3://bucket/key. Recebido: {s3_path}"
        
        bucket_name, object_key = parts
        
        try:
            s3_client.head_object(Bucket=bucket_name, Key=object_key)
        except s3_client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                return f"Erro: Arquivo não encontrado em S3: {s3_path}"
            elif error_code == '403':
                return f"Erro: Sem permissão para acessar {s3_path}. Verifique as credenciais AWS."
            else:
                return f"Erro ao verificar arquivo S3: {str(e)}"
        
        media_format = s3_path.split('.')[-1].lower()
        if media_format not in ['mp3', 'mp4', 'wav', 'flac', 'ogg', 'amr', 'webm']:
            return f"Erro: Formato de áudio '{media_format}' não suportado."
        
        job_params = {
            'TranscriptionJobName': job_name,
            'Media': {'MediaFileUri': s3_path},
            'MediaFormat': media_format,
            'LanguageCode': 'pt-BR',
            'Settings': {
                'ShowSpeakerLabels': False,
                # 'MaxAlternatives': 2
            }
        }
        
        output_bucket = os.getenv("TRANSCRIBE_OUTPUT_BUCKET", bucket_name)
        job_params['Settings']['OutputBucketName'] = output_bucket
        
        transcribe.start_transcription_job(**job_params)
        
        max_wait = 300
        elapsed = 0
        while elapsed < max_wait:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status == 'COMPLETED':
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                with urllib.request.urlopen(transcript_uri) as response:
                    transcript_data = json.loads(response.read().decode())
                    transcript_text = transcript_data['results']['transcripts'][0]['transcript']
                    return transcript_text
            
            elif job_status == 'FAILED':
                failure_reason = status['TranscriptionJob'].get('FailureReason', 'Desconhecido')
                return f"Falha na transcrição: {failure_reason}"
            
            time.sleep(2)
            elapsed += 2
        
        return "Erro: Timeout aguardando transcrição."
        

# ============================================================================
# 3. AGENTES E ORQUESTRAÇÃO
# ============================================================================
def iniciar_analise_multimodal(dados_biometria: Optional[dict] = None, s3_audio: Optional[str] = None):
    # Agente 1: Analista de Dados (Sinais Vitais)
    analyst = Agent(
        role='Analista Biométrico',
        goal='Interpretar sinais vitais via SageMaker.',
        backstory='Especialista em identificar anomalias em dados tabulares de saúde.',
        tools=[HealthTools.predict_risk],
        llm=local_llm,
        allow_delegation=False,
        max_iter=2,
    )

    # Agente 2: Analista de Áudio (Comportamental)
    psychologist = Agent(
        role='Psicólogo Perinatal',
        goal='Detectar sinais de ansiedade ou depressão em áudios.',
        backstory='Especialista em saúde mental materna e análise de discurso.',
        tools=[HealthTools.transcribe_consultation],
        llm=local_llm,
        allow_delegation=False,
        max_iter=2,
    )

    # Agente 3: Médico Chefe (Sintetizador)
    chief = Agent(
        role='Médico Obstetra Sênior',
        goal='Consolidar todas as análises disponíveis em um laudo final.',
        backstory='Responsável pela decisão final, integrando dados técnicos e emocionais.',
        llm=local_llm,
        allow_delegation=False,
        max_iter=2,
    )

    tasks = []
    
    # Adiciona tarefas dinamicamente conforme o que o usuário quer analisar
    if dados_biometria:
        t1 = Task(
            description=f"Analise os dados: {dados_biometria}",
            expected_output="Status de risco técnico (Alto/Baixo).",
            agent=analyst
        )
        tasks.append(t1)

    if s3_audio:
        t2 = Task(
            description=f"Processe o áudio em: {s3_audio}",
            expected_output="Análise qualitativa do estado emocional.",
            agent=psychologist
        )
        tasks.append(t2)

    # Tarefa final sempre consolida o que foi feito
    t_final = Task(
        description="Sintetize as análises anteriores. Se algum dado faltar, baseie-se no que está disponível.",
        expected_output="Relatório final estruturado em JSON.",
        agent=chief,
        context=tasks,
        output_json=RelatorioSaude
    )
    tasks.append(t_final)

    crew = Crew(
        agents=[analyst, psychologist, chief],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()

# ============================================================================
# 4. EXECUÇÃO FLEXÍVEL
# ============================================================================
if __name__ == "__main__":
    # Exemplo 1: Analisar AMBOS (Multimodal completo)
    biometria = {"Age": 35, "SystolicBP": 140, "DiastolicBP": 90, "BS": 13, "BodyTemp": 98, "HeartRate": 70}
    audio_path = "s3://fiap-pos-fase04-matheuslucena/vitima-01.mp3"
    
    print("\n--- INICIANDO ANÁLISE COMPLETA ---")
    resultado = iniciar_analise_multimodal(dados_biometria=biometria, s3_audio=audio_path)
    print(resultado)

    # Exemplo 2: Analisar APENAS Sinais Vitais
    # print("\n--- ANÁLISE APENAS BIOMÉTRICA ---")
    # resultado_so_bio = iniciar_analise_multimodal(dados_biometria=biometria)