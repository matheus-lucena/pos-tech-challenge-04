"""
Task description templates for CrewAI agents.

Keeping prompts in a dedicated module separates them from orchestration logic,
making them easier to read, translate, and iterate on without touching agent code.
"""

import json
from typing import Optional


def biometric_task_description(biometric_data: dict) -> str:
    biometric_json = json.dumps(biometric_data)
    return (
        f"Use the predict_risk tool to analyze the following biometric data: {biometric_data}. "
        f"You can call it as: predict_risk(data_json='{biometric_json}') or simply predict_risk(). "
        f"The biometric data is: Age={biometric_data.get('Age')}, "
        f"SystolicBP={biometric_data.get('SystolicBP')}, "
        f"DiastolicBP={biometric_data.get('DiastolicBP')}, "
        f"BS={biometric_data.get('BS')}, "
        f"BodyTemp={biometric_data.get('BodyTemp')}, "
        f"HeartRate={biometric_data.get('HeartRate')}."
    )


def audio_task_description(s3_audio: str) -> str:
    return f"Process consultation audio at: {s3_audio}"


def maternal_task_description(
    s3_maternal_audio: str,
    biometric_data: Optional[dict] = None,
) -> str:
    heart_rate_info = ""
    if biometric_data and biometric_data.get("HeartRate"):
        heart_rate_info = (
            f"\n\nIMPORTANTE: Os dados biométricos fornecidos incluem Frequência Cardíaca = "
            f"{biometric_data.get('HeartRate')} bpm. "
            f"Use este valor como referência na análise materna. NÃO invente valores de bpm. "
            f"Se o sinal PCG detectar um MHR diferente, mencione ambos os valores "
            f"(o do PCG e o dos dados biométricos)."
        )

    return (
        f"Analyze maternal heart signal (PCG) at: {s3_maternal_audio}. "
        f"Use analyze_maternal_heart_sound tool with is_s3_path=True.{heart_rate_info}"
    )


def final_task_description(biometric_data: Optional[dict] = None) -> str:
    heart_rate_note = ""
    if biometric_data and biometric_data.get("HeartRate"):
        heart_rate_note = (
            f" The 'maternal_analysis' MUST use the HeartRate from biometric data "
            f"({biometric_data.get('HeartRate')} bpm), not invented values. "
        )

    sagemaker_note = ""
    if biometric_data:
        sagemaker_note = (
            "\n\nCRITICAL: The SageMaker model analysis (from the Biometric Analyst task) is the "
            "PRIMARY risk indicator. "
            "If the SageMaker model returned 'HIGH RISK', the 'maternal_analysis' MUST reflect this "
            "HIGH RISK status, even if the PCG signal analysis shows normal values. "
            "The 'maternal_analysis' should integrate BOTH the SageMaker risk assessment AND the PCG "
            "signal analysis. "
            "If there's a conflict, prioritize the SageMaker result as it analyzes comprehensive "
            "biometric data. "
            "The 'final_risk' MUST match the SageMaker risk assessment (HIGH RISK or LOW RISK)."
        )

    return (
        "Synthesize all previous analyses (biometric and maternal) into a final report in PORTUGUESE "
        "(Brazilian Portuguese). "
        "NOTE: Emotional analysis is NOT included in this report - it is only used for separate "
        "consultation audio analysis. "
        "If any data is missing, base on what is available. "
        "Especially integrate maternal analysis with biometric data AND SageMaker risk assessment "
        "for a complete view. "
        "\n\nIMPORTANT: "
        "- All text content MUST be in PORTUGUESE (Brazilian Portuguese). "
        "- The 'biometric_analysis' and 'maternal_analysis' fields must be written in PORTUGUESE. "
        "- The 'emotional_analysis' field should be set to "
        "'Não aplicável - análise emocional é apenas para áudio de consulta separado' (not applicable). "
        f"-{heart_rate_note}"
        f"-{sagemaker_note}"
        "- The 'maternal_analysis' MUST integrate the SageMaker risk assessment result. If SageMaker "
        "indicates HIGH RISK, the maternal analysis must acknowledge this and explain the risk factors, "
        "even if PCG signal appears normal. "
        "- The 'final_risk' field MUST match the SageMaker risk assessment: if SageMaker returned "
        "HIGH RISK, use 'ALTO RISCO'; if LOW RISK, use 'BAIXO RISCO'. "
        "- The 'recommendations' array MUST contain ONLY recommendations (suggestions, guidance), "
        "NOT medical procedures or treatments. "
        "- Recommendations should be general guidance, NOT specific medical procedures like "
        "'administer medication' or 'perform surgery'. "
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
    )


def final_task_expected_output() -> str:
    return (
        "Final structured report in JSON format, with ALL content in PORTUGUESE (Brazilian "
        "Portuguese), including biometric_analysis, emotional_analysis (set to 'Não aplicável'), "
        "maternal_analysis (integrating SageMaker risk assessment and PCG analysis), final_risk "
        "(matching SageMaker result), and recommendations array (guidance only, no medical procedures)."
    )
