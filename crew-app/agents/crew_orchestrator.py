"""Orquestração dos agentes CrewAI para análise multimodal."""

from typing import List, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from models.relatorio_saude import RelatorioSaude
from tools.health_tools import predict_risk, transcribe_consultation


def criar_agentes(llm: LLM) -> Tuple[Agent, Agent, Agent]:
    """
    Cria os agentes necessários para a análise.
    
    Args:
        llm: Instância do LLM configurado
    
    Returns:
        Tupla com (analyst, psychologist, chief)
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

    # Agente 3: Médico Chefe (Sintetizador)
    chief = Agent(
        role='Médico Obstetra Sênior',
        goal='Consolidar todas as análises disponíveis em um laudo final.',
        backstory='Responsável pela decisão final, integrando dados técnicos e emocionais.',
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )
    
    return analyst, psychologist, chief


def criar_tarefas(
    analyst: Agent,
    psychologist: Agent,
    chief: Agent,
    dados_biometria: Optional[dict] = None,
    s3_audio: Optional[str] = None
) -> List[Task]:
    """
    Cria as tarefas para os agentes baseado nos dados disponíveis.
    
    Args:
        analyst: Agente analista biométrico
        psychologist: Agente psicólogo
        chief: Agente médico chefe
        dados_biometria: Dados biométricos opcionais
        s3_audio: Caminho S3 do áudio opcional
    
    Returns:
        Lista de tarefas
    """
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
        description=(
            "Sintetize as análises anteriores. "
            "Se algum dado faltar, baseie-se no que está disponível."
        ),
        expected_output="Relatório final estruturado em JSON.",
        agent=chief,
        context=tasks,
        output_json=RelatorioSaude
    )
    tasks.append(t_final)
    
    return tasks


def iniciar_analise_multimodal(
    llm: LLM,
    dados_biometria: Optional[dict] = None,
    s3_audio: Optional[str] = None
):
    """
    Inicia a análise multimodal usando agentes CrewAI.
    
    Args:
        llm: Instância do LLM configurado
        dados_biometria: Dados biométricos opcionais
        s3_audio: Caminho S3 do áudio opcional
    
    Returns:
        Resultado da execução do crew
    """
    # Cria os agentes
    analyst, psychologist, chief = criar_agentes(llm)
    
    # Cria as tarefas
    tasks = criar_tarefas(analyst, psychologist, chief, dados_biometria, s3_audio)
    
    # Cria e executa o crew
    crew = Crew(
        agents=[analyst, psychologist, chief],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()

