import os
from crewai import LLM


def get_llm() -> LLM:
    os.environ["OPENAI_API_KEY"] = "sk-ollama-local"
    return LLM(
        model="ollama/qwen2.5:7b",
        base_url="http://localhost:11434"
    )

