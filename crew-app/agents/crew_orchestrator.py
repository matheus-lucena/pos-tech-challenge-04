"""Orquestração dos agentes CrewAI para análise multimodal."""

from typing import List, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from models.relatorio_saude import RelatorioSaude
from tools.health_tools import predict_risk, transcribe_consultation
from tools.fetal_tools import analyze_fetal_heart_sound


def criar_agentes(llm: LLM) -> Tuple[Agent, Agent, Agent, Agent]:
    """
    Cria os agentes necessários para a análise.
    
    Args:
        llm: Instância do LLM configurado
    
    Returns:
        Tupla com (analyst, psychologist, fetal_analyst, chief)
    """
    # Agente 1: Analista de Dados (Sinais Vitais)
    analyst = Agent(
        role='Analista Biométrico',
        goal='Interpretar sinais vitais via SageMaker.',
        backstory='Especialista em identificar anomalias em dados tabulares de saúde.',
        tools=[predict_risk],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    # Agente 2: Analista de Áudio (Comportamental)
    psychologist = Agent(
        role='Psicólogo Perinatal',
        goal='Detectar sinais de ansiedade ou depressão em áudios.',
        backstory='Especialista em saúde mental materna e análise de discurso.',
        tools=[transcribe_consultation],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    # Agente 3: Analista de Sinais Fetais
    fetal_analyst = Agent(
        role='Especialista em Monitoramento Fetal',
        goal='Analisar sinais de coração fetal (PCG) e detectar anomalias na frequência cardíaca fetal (FHR).',
        backstory='Especialista em análise de fonocardiogramas fetais, detecção de bradicardia, taquicardia e variabilidade da FHR. Experiente com banco de dados SUFHSDB.',
        tools=[analyze_fetal_heart_sound],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    # Agente 4: Médico Chefe (Sintetizador)
    chief = Agent(
        role='Médico Obstetra Sênior',
        goal='Consolidar todas as análises disponíveis (biométricos, emocional e fetal) em um laudo final.',
        backstory='Responsável pela decisão final, integrando dados técnicos, emocionais e monitoramento fetal.',
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )
    
    return analyst, psychologist, fetal_analyst, chief


def criar_tarefas(
    analyst: Agent,
    psychologist: Agent,
    fetal_analyst: Agent,
    chief: Agent,
    dados_biometria: Optional[dict] = None,
    s3_audio: Optional[str] = None,
    s3_fetal_audio: Optional[str] = None
) -> List[Task]:
    """
    Cria as tarefas para os agentes baseado nos dados disponíveis.
    
    Args:
        analyst: Agente analista biométrico
        psychologist: Agente psicólogo
        fetal_analyst: Agente analista de sinais fetais
        chief: Agente médico chefe
        dados_biometria: Dados biométricos opcionais
        s3_audio: Caminho S3 do áudio opcional (consulta/emocional)
        s3_fetal_audio: Caminho S3 do áudio fetal (PCG) opcional
    
    Returns:
        Lista de tarefas
    """
    tasks = []
    
    # Adiciona tarefas dinamicamente conforme o que o usuário quer analisar
    if dados_biometria:
        t1 = Task(
            description=f"Analise os dados biométricos: {dados_biometria}",
            expected_output="Status de risco técnico (Alto/Baixo) baseado em sinais vitais maternos.",
            agent=analyst
        )
        tasks.append(t1)

    if s3_audio:
        t2 = Task(
            description=f"Processe o áudio de consulta em: {s3_audio}",
            expected_output="Análise qualitativa do estado emocional e psicológico da paciente.",
            agent=psychologist
        )
        tasks.append(t2)

    if s3_fetal_audio:
        t3 = Task(
            description=f"Analise o sinal de coração fetal (PCG) em: {s3_fetal_audio}. Use a ferramenta analyze_fetal_heart_sound com is_s3_path=True.",
            expected_output="Análise completa do sinal fetal incluindo FHR, variabilidade, classificação de risco e recomendações.",
            agent=fetal_analyst
        )
        tasks.append(t3)

    # Tarefa final sempre consolida o que foi feito
    t_final = Task(
        description=(
            "Sintetize todas as análises anteriores (biométrica, emocional e fetal). "
            "Se algum dado faltar, baseie-se no que está disponível. "
            "Integre especialmente a análise fetal com os outros dados para uma visão completa."
        ),
        expected_output="Relatório final estruturado em JSON incluindo análise_fetal.",
        agent=chief,
        context=tasks,
        output_json=RelatorioSaude
    )
    tasks.append(t_final)
    
    return tasks


def iniciar_analise_multimodal(
    llm: LLM,
    dados_biometria: Optional[dict] = None,
    s3_audio: Optional[str] = None,
    s3_fetal_audio: Optional[str] = None
):
    """
    Inicia a análise multimodal usando agentes CrewAI.
    
    Args:
        llm: Instância do LLM configurado
        dados_biometria: Dados biométricos opcionais
        s3_audio: Caminho S3 do áudio opcional (consulta/emocional)
        s3_fetal_audio: Caminho S3 do áudio fetal (PCG) opcional
    
    Returns:
        Resultado da execução do crew
    """
    # Cria os agentes
    analyst, psychologist, fetal_analyst, chief = criar_agentes(llm)
    
    # Cria as tarefas
    tasks = criar_tarefas(
        analyst, psychologist, fetal_analyst, chief,
        dados_biometria, s3_audio, s3_fetal_audio
    )
    
    # Cria e executa o crew
    crew = Crew(
        agents=[analyst, psychologist, fetal_analyst, chief],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()

