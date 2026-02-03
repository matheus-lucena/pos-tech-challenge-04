"""Configuração do LLM para o CrewAI."""

import os
from crewai import LLM


def get_llm() -> LLM:
    """
    Retorna a instância configurada do LLM.
    
    Returns:
        Instância do LLM configurada
    """
    os.environ["OPENAI_API_KEY"] = "sk-ollama-local"
    
    return LLM(
        model="ollama/qwen2.5:7b",
        base_url="http://localhost:11434"
    )

