from typing import List, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from models.relatorio_saude import HealthReport
from tools.health_tools import predict_risk, transcribe_consultation
from tools.maternal_tools import analyze_maternal_heart_sound


def create_agents(llm: LLM) -> Tuple[Agent, Agent, Agent, Agent]:
    analyst = Agent(
        role='Biometric Analyst',
        goal='Interpret vital signs via SageMaker.',
        backstory='Specialist in identifying anomalies in tabular health data.',
        tools=[predict_risk],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    psychologist = Agent(
        role='Perinatal Psychologist',
        goal='Detect signs of anxiety or depression in audio.',
        backstory='Specialist in maternal mental health and discourse analysis.',
        tools=[transcribe_consultation],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    maternal_analyst = Agent(
        role='Maternal Monitoring Specialist',
        goal='Analyze maternal heart signals (PCG) and detect anomalies in maternal heart rate (MHR).',
        backstory='Specialist in maternal phonocardiogram analysis and MHR variability assessment for resting pregnant patients.',
        tools=[analyze_maternal_heart_sound],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    chief = Agent(
        role='Senior Obstetrician',
        goal='Consolidate all available analyses (biometric, emotional and maternal) into a final report.',
        backstory='Responsible for final decision, integrating technical, emotional and maternal monitoring data.',
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )
    
    return analyst, psychologist, maternal_analyst, chief


def create_tasks(
    analyst: Agent,
    psychologist: Agent,
    maternal_analyst: Agent,
    chief: Agent,
    biometric_data: Optional[dict] = None,
    s3_audio: Optional[str] = None,
    s3_maternal_audio: Optional[str] = None
) -> List[Task]:
    tasks = []
    
    if biometric_data:
        t1 = Task(
            description=f"Analyze biometric data: {biometric_data}",
            expected_output="Technical risk status (High/Low) based on maternal vital signs.",
            agent=analyst
        )
        tasks.append(t1)

    if s3_audio:
        t2 = Task(
            description=f"Process consultation audio at: {s3_audio}",
            expected_output="Qualitative analysis of patient's emotional and psychological state.",
            agent=psychologist
        )
        tasks.append(t2)

    if s3_maternal_audio:
        t3 = Task(
            description=f"Analyze maternal heart signal (PCG) at: {s3_maternal_audio}. Use analyze_maternal_heart_sound tool with is_s3_path=True.",
            expected_output="Complete maternal signal analysis including MHR, variability, risk classification and recommendations.",
            agent=maternal_analyst
        )
        tasks.append(t3)

    t_final = Task(
        description=(
            "Synthesize all previous analyses (biometric, emotional and maternal). "
            "If any data is missing, base on what is available. "
            "Especially integrate maternal analysis with other data for a complete view."
        ),
        expected_output="Final structured report in JSON including maternal_analysis.",
        agent=chief,
        context=tasks,
        output_json=HealthReport
    )
    tasks.append(t_final)
    
    return tasks


def start_multimodal_analysis(
    llm: LLM,
    biometric_data: Optional[dict] = None,
    s3_audio: Optional[str] = None,
    s3_maternal_audio: Optional[str] = None
):
    analyst, psychologist, maternal_analyst, chief = create_agents(llm)
    
    tasks = create_tasks(
        analyst, psychologist, maternal_analyst, chief,
        biometric_data, s3_audio, s3_maternal_audio
    )
    
    crew = Crew(
        agents=[analyst, psychologist, maternal_analyst, chief],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()
