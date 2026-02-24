# Maternal Health Risk — Modelo de ML

Modelo XGBoost para classificação de risco gestacional, treinado e implantado no AWS SageMaker.

## Entradas e Saída

**Entrada:**

| Campo | Descrição |
|---|---|
| `Age` | Idade da paciente |
| `SystolicBP` | Pressão arterial sistólica |
| `DiastolicBP` | Pressão arterial diastólica |
| `BS` | Glicemia (Blood Sugar) |
| `BodyTemp` | Temperatura corporal |
| `HeartRate` | Frequência cardíaca |

**Saída:**

```json
{
  "maternal_health_risk": false,
  "risk_probability": "15.23%"
}
```

Classificação binária: `false` = Baixo Risco, `true` = Alto Risco.

---

## Fluxo

```
Dataset CSV → S3 → SageMaker Training Job → Model Artifact → SageMaker Serverless Endpoint
```

---

## Estrutura

```
maternal-health-risk/
├── code/
│   ├── train.py                    # Treinamento (executado no SageMaker)
│   ├── inference.py                # Inferência (executado no SageMaker)
│   └── maternal_health_risk.csv    # Dataset de treinamento
├── deploy.py                       # Deploy completo (upload + treino + endpoint)
├── demo.py                         # Testes rápidos contra o endpoint
└── iam-user-permissions.md         # Permissões IAM necessárias
```

---

## Configuração

Crie `.env` na raiz de `maternal-health-risk/`:

```env
AWS_ROLE_SAGEMAKER=arn:aws:iam::ACCOUNT_ID:role/maternal-health-system-sagemaker-role
AWS_SAGEMAKER_BUCKET=seu-bucket-sagemaker
AWS_DEFAULT_REGION=us-east-1
```

A role e o bucket devem ser criados pelo módulo Terraform em `infra/`.

---

## Uso

### Deploy

```bash
python deploy.py
```

Faz upload do CSV para o S3, inicia o job de treinamento e cria o endpoint serverless. Tempo estimado: 10–15 minutos.

### Testar o endpoint

```bash
python demo.py
```

Ou diretamente via SDK:

```python
import sagemaker, json
from sagemaker.predictor import Predictor

predictor = Predictor(
    endpoint_name="maternal-risk-xgb-...",
    sagemaker_session=sagemaker.Session()
)

data = {
    "Age": 25, "SystolicBP": 120, "DiastolicBP": 80,
    "BS": 7.5, "BodyTemp": 98, "HeartRate": 70
}
response = predictor.predict(json.dumps(data))
print(response)
```

### Dataset

O arquivo `code/maternal_health_risk.csv` deve ter o formato:

```csv
Age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate,RiskLevel
25,130,80,15,98,86,high risk
35,140,90,13,98,70,high risk
```

---

## Hiperparâmetros

Configuráveis em `deploy.py`:

```python
hyperparameters = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1
}
```

---

## Custos

| Recurso | Custo |
|---|---|
| Training Job (ml.m5.large) | ~$0.115/hora (~$0.02 por execução) |
| Serverless Endpoint (2048MB) | ~$0.000004/ms — cobrado por invocação |

Delete o endpoint quando não estiver em uso:

```bash
aws sagemaker delete-endpoint --endpoint-name seu-endpoint-name
aws sagemaker delete-endpoint-config --endpoint-config-name seu-endpoint-config-name
aws sagemaker delete-model --model-name seu-model-name
```

---

## Troubleshooting

**Role não encontrada** — verifique se foi criada pelo Terraform e se o ARN está correto no `.env`.

**Bucket não encontrado** — confirme o nome do bucket no `.env` e se o recurso existe na AWS.

**Timeout no treinamento** — consulte os logs no CloudWatch para detalhes do erro.

**Endpoint não responde** — verifique se o status está `InService` no console da AWS.

---

## Integração

O endpoint é invocado pelo `app` via `services/sagemaker_service.py` para predições a partir dos dados biométricos inseridos na interface.

---

## Referências

- [AWS SageMaker XGBoost Container](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
