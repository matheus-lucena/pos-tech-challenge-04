import json
from typing import List, Optional, Tuple
from crewai import Agent, Task, Crew, Process, LLM
from models.report import HealthReport
from tools.health_tools import predict_risk, transcribe_consultation, set_biometric_data
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
        role='Obstetra Sênior',
        goal='Consolidar todas as análises disponíveis (biométrica, emocional e materna) em um relatório final em PORTUGUÊS (português brasileiro).',
        backstory='Responsável pela decisão final, integrando dados técnicos, emocionais e de monitoramento materno. Especialista em saúde materna que sempre produz relatórios e recomendações em português brasileiro para facilitar a comunicação com a equipe médica brasileira.',
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
        set_biometric_data(biometric_data)
        
        biometric_json = json.dumps(biometric_data)
        t1 = Task(
            description=(
                f"Use the predict_risk tool to analyze the following biometric data: {biometric_data}. "
                f"You can call it as: predict_risk(data_json='{biometric_json}') or simply predict_risk(). "
                f"The biometric data is: Age={biometric_data.get('Age')}, "
                f"SystolicBP={biometric_data.get('SystolicBP')}, "
                f"DiastolicBP={biometric_data.get('DiastolicBP')}, "
                f"BS={biometric_data.get('BS')}, "
                f"BodyTemp={biometric_data.get('BodyTemp')}, "
                f"HeartRate={biometric_data.get('HeartRate')}."
            ),
            expected_output="Technical risk status (HIGH RISK or LOW RISK) with confidence probability based on maternal vital signs.",
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
        heart_rate_info = ""
        if biometric_data and biometric_data.get('HeartRate'):
            heart_rate_info = (
                f"\n\nIMPORTANTE: Os dados biométricos fornecidos incluem Frequência Cardíaca = {biometric_data.get('HeartRate')} bpm. "
                f"Use este valor como referência na análise materna. NÃO invente valores de bpm. "
                f"Se o sinal PCG detectar um MHR diferente, mencione ambos os valores (o do PCG e o dos dados biométricos)."
            )
        
        t3 = Task(
            description=(
                f"Analyze maternal heart signal (PCG) at: {s3_maternal_audio}. "
                f"Use analyze_maternal_heart_sound tool with is_s3_path=True.{heart_rate_info}"
            ),
            expected_output="Complete maternal signal analysis including MHR, variability, risk classification and recommendations. MUST use HeartRate from biometric data if available.",
            agent=maternal_analyst
        )
        tasks.append(t3)

    heart_rate_note = ""
    if biometric_data and biometric_data.get('HeartRate'):
        heart_rate_note = f" The 'maternal_analysis' MUST use the HeartRate from biometric data ({biometric_data.get('HeartRate')} bpm), not invented values. "
    
    sage_maker_note = ""
    if biometric_data:
        sage_maker_note = (
            "\n\nCRITICAL: The SageMaker model analysis (from the Biometric Analyst task) is the PRIMARY risk indicator. "
            "If the SageMaker model returned 'HIGH RISK', the 'maternal_analysis' MUST reflect this HIGH RISK status, "
            "even if the PCG signal analysis shows normal values. "
            "The 'maternal_analysis' should integrate BOTH the SageMaker risk assessment AND the PCG signal analysis. "
            "If there's a conflict, prioritize the SageMaker result as it analyzes comprehensive biometric data. "
            "The 'final_risk' MUST match the SageMaker risk assessment (HIGH RISK or LOW RISK)."
        )
    
    t_final = Task(
        description=(
            "Synthesize all previous analyses (biometric and maternal) into a final report in PORTUGUESE (Brazilian Portuguese). "
            "NOTE: Emotional analysis is NOT included in this report - it is only used for separate consultation audio analysis. "
            "If any data is missing, base on what is available. "
            "Especially integrate maternal analysis with biometric data AND SageMaker risk assessment for a complete view. "
            "\n\nIMPORTANT: "
            "- All text content MUST be in PORTUGUESE (Brazilian Portuguese). "
            "- The 'biometric_analysis' and 'maternal_analysis' fields must be written in PORTUGUESE. "
            "- The 'emotional_analysis' field should be set to 'Não aplicável - análise emocional é apenas para áudio de consulta separado' (not applicable). "
            f"-{heart_rate_note}"
            f"-{sage_maker_note}"
            "- The 'maternal_analysis' MUST integrate the SageMaker risk assessment result. If SageMaker indicates HIGH RISK, "
            "the maternal analysis must acknowledge this and explain the risk factors, even if PCG signal appears normal. "
            "- The 'final_risk' field MUST match the SageMaker risk assessment: if SageMaker returned HIGH RISK, use 'ALTO RISCO'; if LOW RISK, use 'BAIXO RISCO'. "
            "- The 'recommendations' array MUST contain ONLY recommendations (suggestions, guidance), NOT medical procedures or treatments. "
            "- Recommendations should be general guidance, NOT specific medical procedures like 'administer medication' or 'perform surgery'. "
            "- Use medical terminology in Portuguese. "
            "\n\nExample of recommendations in Portuguese (guidance only, NOT procedures): "
            "- 'Monitoramento contínuo recomendado' "
            "- 'Avaliação médica imediata recomendada' "
            "- 'Observação cuidadosa dos sinais vitais' "
            "- 'Acompanhamento próximo da gestante' "
            "\n\nDO NOT include medical procedures like: "
            "- 'Administrar medicamento X' "
            "- 'Realizar cirurgia' "
            "- 'Aplicar tratamento Y'"
        ),
        expected_output="Final structured report in JSON format, with ALL content in PORTUGUESE (Brazilian Portuguese), including biometric_analysis, emotional_analysis (set to 'Não aplicável'), maternal_analysis (integrating SageMaker risk assessment and PCG analysis), final_risk (matching SageMaker result), and recommendations array (guidance only, no medical procedures).",
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
