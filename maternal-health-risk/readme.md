# 🤰 Maternal Health Risk Prediction Model

Modelo de Machine Learning para predição de risco de saúde materna usando XGBoost, treinado e implantado no AWS SageMaker.

## 📋 Descrição

Este projeto implementa um modelo de classificação binária que prediz o risco de saúde materna com base em dados biométricos:

- **Idade** (Age)
- **Pressão Arterial Sistólica** (SystolicBP)
- **Pressão Arterial Diastólica** (DiastolicBP)
- **Glicemia** (BS - Blood Sugar)
- **Temperatura Corporal** (BodyTemp)
- **Frequência Cardíaca** (HeartRate)

O modelo utiliza **XGBoost** e é treinado no **AWS SageMaker**, com deploy em endpoint serverless para inferência.

## 🎯 Objetivo

Classificar pacientes gestantes em:
- **Baixo Risco** (Low Risk): `risk = False`
- **Alto Risco** (High Risk): `risk = True`

## 🏗️ Arquitetura

```
Dataset CSV → S3 → SageMaker Training Job → Model Artifact → SageMaker Endpoint (Serverless)
```

### Componentes

1. **Script de Treinamento** (`code/train.py`):
   - Carrega dados do S3
   - Pré-processa os dados
   - Treina modelo XGBoost
   - Salva modelo no formato joblib

2. **Script de Inferência** (`code/inference.py`):
   - Carrega modelo treinado
   - Processa requisições JSON
   - Retorna predição e probabilidade

3. **Script de Deploy** (`deploy.py`):
   - Faz upload do dataset para S3
   - Cria job de treinamento no SageMaker
   - Faz deploy do modelo em endpoint serverless

## 📦 Pré-requisitos

### Software

- Python 3.8+
- AWS CLI configurado
- Credenciais AWS com permissões para SageMaker

### Dependências

```bash
pip install -r requirements.txt
```

As principais dependências são:
- `sagemaker>=2.*`
- `pandas`
- `xgboost`
- `scikit-learn`
- `joblib`
- `boto3`
- `python-dotenv`

## ⚙️ Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# IAM Role para SageMaker (criada pelo Terraform)
AWS_ROLE_SAGEMAKER=arn:aws:iam::ACCOUNT_ID:role/maternal-health-system-sagemaker-role

# Bucket S3 para dados (criado pelo Terraform)
AWS_SAGEMAKER_BUCKET=seu-bucket-sagemaker

# Região AWS
AWS_DEFAULT_REGION=us-east-1
```

**Nota**: A role e o bucket devem ser criados pelo módulo Terraform em `infra/`.

## 🚀 Uso

### 1. Preparar Dados

Certifique-se de que o arquivo `code/maternal_health_risk.csv` existe e está no formato correto:

```csv
Age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate,RiskLevel
25,130,80,15,98,86,high risk
35,140,90,13,98,70,high risk
...
```

### 2. Treinar e Fazer Deploy

Execute o script de deploy:

```bash
python deploy.py
```

Este script:
1. Faz upload do dataset para S3
2. Inicia um job de treinamento no SageMaker
3. Aguarda o treinamento completar
4. Faz deploy do modelo em um endpoint serverless

**Tempo estimado**: 10-15 minutos (dependendo da região AWS)

### 3. Testar o Modelo

Após o deploy, você pode testar usando o script de demo:

```bash
python demo.py
```

Ou usar diretamente o SageMaker Predictor:

```python
import sagemaker
from sagemaker.predictor import Predictor
import json

# Substitua pelo nome do seu endpoint
endpoint_name = "maternal-risk-xgb-..."

predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker.Session()
)

# Dados de exemplo
data = {
    "Age": 25,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 7.5,
    "BodyTemp": 98,
    "HeartRate": 70
}

# Fazer predição
response = predictor.predict(json.dumps(data))
print(response)
```

### 4. Formato de Entrada e Saída

**Entrada (JSON)**:
```json
{
  "Age": 25,
  "SystolicBP": 120,
  "DiastolicBP": 80,
  "BS": 7.5,
  "BodyTemp": 98,
  "HeartRate": 70
}
```

**Saída (JSON)**:
```json
{
  "maternal_health_risk": false,
  "risk_probability": "15.23%"
}
```

## 📁 Estrutura do Projeto

```
maternal-health-risk/
├── code/
│   ├── train.py              # Script de treinamento (SageMaker)
│   ├── inference.py           # Script de inferência (SageMaker)
│   └── maternal_health_risk.csv  # Dataset de treinamento
├── deploy.py                  # Script de deploy completo
├── demo.py                    # Script de demonstração
├── requirements.txt           # Dependências Python
├── iam-user-permissions.md    # Documentação de permissões IAM
└── README.md                  # Este arquivo
```

## 🔧 Hiperparâmetros

O modelo XGBoost é treinado com os seguintes hiperparâmetros (configuráveis em `deploy.py`):

```python
hyperparameters = {
    'n_estimators': 100,      # Número de árvores
    'max_depth': 5,            # Profundidade máxima
    'learning_rate': 0.1       # Taxa de aprendizado
}
```

## 📊 Métricas de Avaliação

O script de treinamento calcula:
- **Accuracy**: Acurácia do modelo
- **Classification Report**: Relatório detalhado com precision, recall e F1-score

## 💰 Custos

### SageMaker Training
- **Instance Type**: `ml.m5.large`
- **Custo aproximado**: ~$0.115/hora
- **Tempo de treinamento**: ~5-10 minutos

### SageMaker Endpoint (Serverless)
- **Memory**: 2048 MB
- **Max Concurrency**: 5
- **Custo**: Pay-per-use (cobrado apenas quando invocado)
- **Custo aproximado**: ~$0.000004/ms de execução

**Dica**: Delete o endpoint quando não estiver em uso para evitar custos.

## 🗑️ Limpeza de Recursos

Para evitar custos desnecessários, delete os recursos após o uso:

```python
import sagemaker

sess = sagemaker.Session()
predictor = sagemaker.predictor.Predictor(
    endpoint_name="seu-endpoint-name",
    sagemaker_session=sess
)

# Deletar endpoint
predictor.delete_endpoint()
```

Ou via AWS CLI:

```bash
aws sagemaker delete-endpoint --endpoint-name seu-endpoint-name
aws sagemaker delete-endpoint-config --endpoint-config-name seu-endpoint-config-name
aws sagemaker delete-model --model-name seu-model-name
```

## 🐛 Troubleshooting

### Erro: Role não encontrada
- Verifique se a role foi criada pelo Terraform
- Confirme o ARN da role nas variáveis de ambiente

### Erro: Bucket não encontrado
- Verifique se o bucket foi criado pelo Terraform
- Confirme o nome do bucket nas variáveis de ambiente

### Erro: Timeout no treinamento
- Verifique os logs no CloudWatch
- Aumente o timeout ou verifique a instância

### Erro: Endpoint não responde
- Verifique se o endpoint está "InService"
- Confira os logs do endpoint no CloudWatch

## 📝 Notas Importantes

1. **Dataset**: O arquivo CSV deve estar na pasta `code/` antes de executar `deploy.py`
2. **Permissões**: O usuário/role precisa de permissões para SageMaker, S3 e IAM (PassRole)
3. **Região**: Certifique-se de que todos os recursos estão na mesma região AWS
4. **Custos**: Monitore os custos na AWS Console, especialmente para endpoints serverless

## 🔗 Integração com o App

Este modelo é usado pelo módulo `app` através do serviço `sagemaker_service.py`. O endpoint SageMaker é invocado para fazer predições de risco baseadas em dados biométricos.

## 📚 Referências

- [AWS SageMaker XGBoost Container](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📝 Licença

Este projeto faz parte do sistema de saúde materna desenvolvido para o desafio técnico.
