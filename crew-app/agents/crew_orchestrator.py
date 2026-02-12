from typing import List, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from models.relatorio_saude import HealthReport
from tools.health_tools import predict_risk, transcribe_consultation
from tools.fetal_tools import analyze_fetal_heart_sound


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

    fetal_analyst = Agent(
        role='Fetal Monitoring Specialist',
        goal='Analyze fetal heart signals (PCG) and detect anomalies in fetal heart rate (FHR).',
        backstory='Specialist in fetal phonocardiogram analysis, detection of bradycardia, tachycardia and FHR variability. Experienced with SUFHSDB database.',
        tools=[analyze_fetal_heart_sound],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    chief = Agent(
        role='Senior Obstetrician',
        goal='Consolidate all available analyses (biometric, emotional and fetal) into a final report.',
        backstory='Responsible for final decision, integrating technical, emotional and fetal monitoring data.',
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )
    
    return analyst, psychologist, fetal_analyst, chief


def create_tasks(
    analyst: Agent,
    psychologist: Agent,
    fetal_analyst: Agent,
    chief: Agent,
    biometric_data: Optional[dict] = None,
    s3_audio: Optional[str] = None,
    s3_fetal_audio: Optional[str] = None
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

    if s3_fetal_audio:
        t3 = Task(
            description=f"Analyze fetal heart signal (PCG) at: {s3_fetal_audio}. Use analyze_fetal_heart_sound tool with is_s3_path=True.",
            expected_output="Complete fetal signal analysis including FHR, variability, risk classification and recommendations.",
            agent=fetal_analyst
        )
        tasks.append(t3)

    t_final = Task(
        description=(
            "Synthesize all previous analyses (biometric, emotional and fetal). "
            "If any data is missing, base on what is available. "
            "Especially integrate fetal analysis with other data for a complete view."
        ),
        expected_output="Final structured report in JSON including fetal_analysis.",
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
    s3_fetal_audio: Optional[str] = None
):
    analyst, psychologist, fetal_analyst, chief = create_agents(llm)
    
    tasks = create_tasks(
        analyst, psychologist, fetal_analyst, chief,
        biometric_data, s3_audio, s3_fetal_audio
    )
    
    crew = Crew(
        agents=[analyst, psychologist, fetal_analyst, chief],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()
