from typing import List, Optional, Tuple

from crewai import Agent, Task, Crew, Process, LLM

from models.report import HealthReport
from tools.health_tools import predict_risk, transcribe_consultation, set_biometric_data
from tools.maternal_tools import analyze_maternal_heart_sound
from agents.task_templates import (
    biometric_task_description,
    audio_task_description,
    maternal_task_description,
    final_task_description,
    final_task_expected_output,
)


def create_agents(llm: LLM) -> Tuple[Agent, Agent, Agent, Agent]:
    analyst = Agent(
        role="Biometric Analyst",
        goal="Interpret vital signs via SageMaker.",
        backstory="Specialist in identifying anomalies in tabular health data.",
        tools=[predict_risk],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    psychologist = Agent(
        role="Perinatal Psychologist",
        goal="Detect signs of anxiety or depression in audio.",
        backstory="Specialist in maternal mental health and discourse analysis.",
        tools=[transcribe_consultation],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    maternal_analyst = Agent(
        role="Maternal Monitoring Specialist",
        goal="Analyze maternal heart signals (PCG) and detect anomalies in maternal heart rate (MHR).",
        backstory=(
            "Specialist in maternal phonocardiogram analysis and MHR variability assessment "
            "for resting pregnant patients."
        ),
        tools=[analyze_maternal_heart_sound],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
    )

    chief = Agent(
        role="Senior Obstetrician",
        goal=(
            "Consolidate all available analyses (biometric, emotional, and maternal) into a "
            "final report in Brazilian Portuguese."
        ),
        backstory=(
            "Responsible for the final decision, integrating technical, emotional, and maternal "
            "monitoring data. Maternal health specialist who always produces reports and "
            "recommendations in Brazilian Portuguese for the medical team."
        ),
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
    s3_maternal_audio: Optional[str] = None,
) -> List[Task]:
    tasks = []

    if biometric_data:
        set_biometric_data(biometric_data)
        tasks.append(Task(
            description=biometric_task_description(biometric_data),
            expected_output=(
                "Technical risk status (HIGH RISK or LOW RISK) with confidence probability "
                "based on maternal vital signs."
            ),
            agent=analyst,
        ))

    if s3_audio:
        tasks.append(Task(
            description=audio_task_description(s3_audio),
            expected_output="Qualitative analysis of patient's emotional and psychological state.",
            agent=psychologist,
        ))

    if s3_maternal_audio:
        tasks.append(Task(
            description=maternal_task_description(s3_maternal_audio, biometric_data),
            expected_output=(
                "Complete maternal signal analysis including MHR, variability, risk classification "
                "and recommendations. MUST use HeartRate from biometric data if available."
            ),
            agent=maternal_analyst,
        ))

    tasks.append(Task(
        description=final_task_description(biometric_data),
        expected_output=final_task_expected_output(),
        agent=chief,
        context=tasks,
        output_json=HealthReport,
    ))

    return tasks


def start_multimodal_analysis(
    llm: LLM,
    biometric_data: Optional[dict] = None,
    s3_audio: Optional[str] = None,
    s3_maternal_audio: Optional[str] = None,
):
    analyst, psychologist, maternal_analyst, chief = create_agents(llm)

    tasks = create_tasks(
        analyst, psychologist, maternal_analyst, chief,
        biometric_data, s3_audio, s3_maternal_audio,
    )

    crew = Crew(
        agents=[analyst, psychologist, maternal_analyst, chief],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()
