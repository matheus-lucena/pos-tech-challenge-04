# Sistema Multimodal de Análise de Saúde Materna

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Transcribe%20%7C%20Comprehend-orange)
![Terraform](https://img.shields.io/badge/Terraform-%3E%3D1.0-7B42BC)

Sistema de análise de saúde materna que combina dados biométricos, áudio de consultas e sinais cardíacos para avaliação de risco gestacional.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Gradio (app)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Dados       │  │  Áudio de    │  │  Sinais      │       │
│  │  Biométricos │  │  Consulta    │  │  Cardíacos   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼──────────────────┼─────────────────┼──────────────┘
          │                  │                 │
          ▼                  ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Agentes CrewAI (Orquestração)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Biometric   │  │  Perinatal   │  │  Maternal    │       │
│  │  Analyst     │  │  Psychologist│  │  Monitoring  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         └─────────────────┴─────────────────┘                │
│                            │                                 │
│                            ▼                                 │
│              ┌─────────────────────────┐                     │
│              │  Obstetra Sênior        │                     │
│              │  (Consolidação Final)   │                     │
│              └─────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
          │                  │                 │
          ▼                  ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Serviços AWS                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  SageMaker   │  │  Transcribe  │  │  Comprehend  │       │
│  │  (ML Model)  │  │  (Audio)     │  │  Medical     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         └─────────────────┴─────────────────┘                │
│                            │                                 │
│                            ▼                                 │
│                    ┌──────────────┐                          │
│                    │  S3 Buckets  │                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Módulos

### `infra/` — Infraestrutura AWS (Terraform)

Provisiona os recursos necessários na AWS:

- Buckets S3 para áudios e dados de treinamento
- IAM Roles para SageMaker, Transcribe e demais serviços
- IAM User para execução local
- CloudWatch Log Groups

Documentação: [infra/README.md](infra/README.md)

### `maternal-health-risk/` — Modelo de ML

Modelo XGBoost para predição de risco gestacional:

- Treinamento no AWS SageMaker
- Deploy em endpoint serverless
- Entrada: idade, pressão arterial, glicemia, temperatura, frequência cardíaca

Documentação: [maternal-health-risk/README.md](maternal-health-risk/README.md)

### `app/` — Aplicação Principal

Interface web e orquestração de agentes:

- Interface Gradio
- Agentes CrewAI para análise multimodal
- Transcrição de áudio em tempo real
- Detecção de violência contra mulher

Documentação: [app/README.md](app/README.md)

### `violence-against-women-bert/` — Classificador BERT

BERTimbau fine-tunado para detecção de violência contra mulher em português:

- F1 violence: 0.9337 | Accuracy: 0.9599
- Integrado ao serviço de transcrição em tempo real

Documentação: [violence-against-women-bert/README.md](violence-against-women-bert/README.md)

---

## Início Rápido

### Pré-requisitos

- Python 3.8+
- Terraform >= 1.0
- AWS CLI configurado
- Conta AWS com permissões administrativas

### 1. Provisionar infraestrutura

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
# editar terraform.tfvars com seus valores
terraform init
terraform apply
```

Salve as credenciais do usuário IAM geradas:

```bash
terraform output -raw secret_access_key
```

### 2. Configurar variáveis de ambiente

```bash
cd app
cp .env.example .env
# preencher .env com as credenciais e configurações AWS
```

Variáveis necessárias:

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_ROLE_SAGEMAKER=arn:aws:iam::ACCOUNT:role/maternal-health-system-sagemaker-role
AWS_SAGEMAKER_BUCKET=
AWS_S3_AUDIO_BUCKET=
AWS_TRANSCRIBE_ROLE=arn:aws:iam::ACCOUNT:role/TranscribeDataAccess
LITELLM_API_KEY=
```

### 3. Treinar e fazer deploy do modelo

```bash
cd maternal-health-risk
python deploy.py
```

Aguarde ~10–15 minutos para o treinamento e deploy concluírem.

### 4. Executar a aplicação

```bash
cd app
python app.py
```

Acesse em `http://localhost:7860`.

### (Opcional) Gerar laudos PDF de teste

```bash
python generate_pdf.py           # baixo risco + alto risco + aleatório
python generate_pdf.py --low     # somente baixo risco
python generate_pdf.py --high    # somente alto risco
python generate_pdf.py --random  # somente aleatório
```

| Arquivo | Caso | Idade | PA | Glicemia | Temp |
|---|---|---|---|---|---|
| `laudo_baixo_risco.pdf` | Baixo Risco | 25 anos | 110x70 mmHg | 117 mg/dL | 36.7°C |
| `laudo_alto_risco.pdf` | Alto Risco | 40 anos | 150x100 mmHg | 189 mg/dL | 37.5°C |
| `laudo_medico_exemplo.pdf` | Aleatório | variável | variável | variável | variável |

O PDF usa unidades clínicas brasileiras (mg/dL, °C). O sistema converte para as unidades do modelo (mmol/L, °F) ao processar o arquivo.

---

## Estrutura do Projeto

```
pos-tech-challenge-04/
├── app/                          # Aplicação principal
│   ├── agents/                   # Agentes CrewAI
│   │   └── task_templates.py
│   ├── config/
│   ├── models/
│   ├── services/                 # Serviços AWS
│   ├── tools/                    # Ferramentas dos agentes
│   ├── ui/                       # Interface Gradio + handlers de tempo real
│   ├── utils/
│   └── app.py
│
├── infra/                        # Infraestrutura Terraform
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── terraform.tfvars.example
│
├── maternal-health-risk/         # Modelo XGBoost
│   ├── code/
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── maternal_health_risk.csv
│   ├── deploy.py
│   └── demo.py
│
├── violence-against-women-bert/  # Classificador BERT
│   ├── code/
│   │   ├── train.py
│   │   └── inference.py
│   ├── data/
│   ├── model_output/
│   └── generate_dataset.py
│
├── generate_pdf.py               # Gerador de laudos PDF para testes
└── requirements.txt
```

---

## Tecnologias

| Camada | Tecnologias |
|---|---|
| Interface | Gradio |
| Agentes | CrewAI, LiteLLM |
| ML | XGBoost, BERTimbau, scikit-learn |
| AWS | SageMaker, Transcribe, Comprehend Medical, Textract, S3, CloudWatch |
| Infra | Terraform, IAM |

---

## Fluxo de Dados

1. Usuário insere dados biométricos, áudio e/ou sinais cardíacos na interface
2. Processamento paralelo:
   - Biométricos → SageMaker → predição de risco
   - Áudio → AWS Transcribe → transcrição → análise emocional + detecção de violência
   - Sinais cardíacos → análise PCG → anomalias
3. Agentes CrewAI consolidam as análises
4. Obstetra Sênior gera o relatório final em português

---

## Segurança

- Criptografia AES256 em todos os buckets S3
- Acesso público bloqueado
- Políticas IAM com princípio de menor privilégio
- Logs centralizados no CloudWatch

---

## Custos Estimados

| Serviço | Custo |
|---|---|
| S3 Storage | ~$0.023/GB/mês |
| SageMaker Training (ml.m5.large) | ~$0.115/hora |
| SageMaker Serverless Endpoint | ~$0.000004/ms |
| Transcribe | ~$0.024/minuto |

Delete endpoints e recursos não utilizados para evitar custos desnecessários.

---

## Troubleshooting

**Credenciais AWS inválidas** — verifique as variáveis de ambiente e se a conta tem as permissões necessárias.

**Endpoint SageMaker não encontrado** — confirme que o deploy foi concluído e o nome do endpoint está correto no `.env`.

**Erro de transcrição** — verifique se o bucket S3 existe e se a role do Transcribe tem acesso a ele.

**Bucket S3 não encontrado** — nomes de buckets são únicos globalmente; confirme que foram criados pelo Terraform.
