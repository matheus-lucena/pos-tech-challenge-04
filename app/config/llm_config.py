import os
from crewai import LLM


def get_llm() -> LLM:
    region = os.getenv("AWS_REGION", "us-east-1")
    os.environ["AWS_REGION_NAME"] = region
    return LLM(
        model=os.getenv("LLM_MODEL","us.bedrock/amazon.nova-2-lite-v1:0"),
        temperature=0.1,
    )

