import sagemaker
import os
from sagemaker.estimator import Estimator
from sagemaker import image_uris
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.model import Model
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------
# 1. Configurações Iniciais e Sessão
# ------------------------------------------------------------------------------
print(">>> Configurando sessão SageMaker...")
sess = sagemaker.Session()
region = sess.boto_region_name

role = os.environ.get("AWS_ROLE_SAGEMAKER")
bucket = os.environ.get("AWS_SAGEMAKER_BUCKET")
prefix = 'maternal-health-system/maternal-risk'
sess = sagemaker.Session(default_bucket=bucket)

file_name = 'code/maternal_health_risk.csv'
if not os.path.exists(file_name):
    raise FileNotFoundError(f"O arquivo {file_name} não foi encontrado na pasta atual!")

print(f">>> Enviando {file_name} para o S3...")
train_input = sess.upload_data(
    file_name, 
    bucket=bucket, 
    key_prefix=f'{prefix}/data'
)
print(f"Dados disponíveis em: {train_input}")

# ------------------------------------------------------------------------------
# 3. Recuperação Manual da Imagem Docker (Fix do ModuleNotFoundError)
# ------------------------------------------------------------------------------
container_uri = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1"  # Versão estável do XGBoost
)
print(f">>> Imagem Docker recuperada: {container_uri}")

# ------------------------------------------------------------------------------
# 4. Definição do Estimator
# ------------------------------------------------------------------------------
xgb_estimator = Estimator(
    image_uri=container_uri,
    entry_point='train.py',
    source_dir='code',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    sagemaker_session=sess,
    base_job_name='maternal-risk-xgb',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1
    },
)
# ------------------------------------------------------------------------------
# 5. Treinamento (Na Nuvem)
# ------------------------------------------------------------------------------
print(">>> Iniciando Job de Treinamento na AWS (isso pode levar alguns minutos)...")
xgb_estimator.fit({'train': train_input})
print(">>> Treinamento concluído!")

# ------------------------------------------------------------------------------
# 6. Deploy Serverless
# ------------------------------------------------------------------------------
print(">>> Configurando Deploy Serverless...")

model = Model(
    image_uri=container_uri,
    model_data=xgb_estimator.model_data,
    role=role,
    sagemaker_session=sess,
    entry_point='inference.py',
    source_dir='code',
)

# Configuração Serverless
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5
)

print(">>> Criando Endpoint (Deploy)...")
# Fazemos o deploy do OBJETO MODEL, não do estimator
predictor = model.deploy(
    serverless_inference_config=serverless_config
)

print("-" * 50)
print(f"SUCESSO! Seu endpoint está ativo.")
print(f"Nome do Endpoint: {predictor}")
print("-" * 50)