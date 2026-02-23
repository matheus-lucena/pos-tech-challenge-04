import os
from crewai import LLM
from dotenv import load_dotenv
load_dotenv()


def get_llm() -> LLM:
    """
    Configura o LLM para usar AWS Bedrock com Claude 3 Haiku via LiteLLM.
    
    O LiteLLM já está instalado e tem suporte nativo para Bedrock.
    O CrewAI aceita modelos LiteLLM através de strings no formato correto.
    
    Variáveis de ambiente necessárias:
    - AWS_REGION: região do Bedrock (ex: us-east-1, us-west-2)
    - AWS_ACCESS_KEY_ID: chave de acesso AWS (opcional se usar IAM role/EC2)
    - AWS_SECRET_ACCESS_KEY: chave secreta AWS (opcional se usar IAM role/EC2)
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    os.environ["AWS_REGION_NAME"] = region
    return LLM(
        model=os.getenv("LLM_MODEL","us.bedrock/amazon.nova-2-lite-v1:0"),
        temperature=0.1,
    )

