# 🏥 Sistema Multimodal de Análise de Saúde Materna

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Transcribe%20%7C%20Comprehend-orange)
![Terraform](https://img.shields.io/badge/Terraform-%3E%3D1.0-7B42BC)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

Sistema completo de inteligência artificial para análise de saúde materna, integrando múltiplas fontes de dados (biométricos, áudio, sinais cardíacos) para fornecer uma avaliação abrangente do risco de saúde materna.

## 📋 Visão Geral

Este projeto é uma solução end-to-end que combina:

- **Análise Biométrica**: Predição de risco baseada em sinais vitais usando Machine Learning
- **Análise de Áudio**: Transcrição e análise emocional de consultas médicas
- **Análise de Sinais Cardíacos**: Detecção de anomalias em fonocardiogramas maternos
- **Agentes de IA**: Sistema multi-agente usando CrewAI para análise integrada
- **Interface Web**: Interface interativa construída com Gradio

## 🏗️ Arquitetura do Sistema

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
│         │                 │                 │                │
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
│         │                 │                 │                │
│         └─────────────────┴─────────────────┘                │
│                            │                                 │
│                            ▼                                 │
│                    ┌──────────────┐                          │
│                    │  S3 Buckets  │                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Módulos do Projeto

### 1. 🏗️ `infra/` - Infraestrutura AWS (Terraform)

Provisiona toda a infraestrutura necessária na AWS:

- **Buckets S3**: Armazenamento de áudios e dados de treinamento
- **IAM Roles**: Permissões para SageMaker, Transcribe e outros serviços
- **IAM User**: Usuário para execução local
- **CloudWatch Log Groups**: Logs centralizados

**📖 Documentação**: Veja [infra/README.md](infra/README.md)

### 2. 🤰 `maternal-health-risk/` - Modelo de ML

Modelo XGBoost para predição de risco de saúde materna:

- Treinamento no AWS SageMaker
- Deploy em endpoint serverless
- Predição baseada em dados biométricos (idade, pressão arterial, glicemia, etc.)

**📖 Documentação**: Veja [maternal-health-risk/README.md](maternal-health-risk/README.md)

### 3. 🖥️ `app/` - Aplicação Principal

Interface web e orquestração de agentes:

- Interface Gradio para interação
- Agentes CrewAI para análise multimodal
- Integração com serviços AWS
- Processamento de áudio em tempo real

**📖 Documentação**: Veja [app/README.md](app/README.md)

## 🚀 Início Rápido

### Pré-requisitos

- Python 3.8+
- Terraform >= 1.0
- AWS CLI configurado
- Conta AWS com permissões administrativas
- Credenciais AWS configuradas

### Passo 1: Provisionar Infraestrutura

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
# Edite terraform.tfvars com seus valores
terraform init
terraform apply
```

**Importante**: Salve as credenciais do usuário IAM geradas:
```bash
terraform output -raw secret_access_key
```

### Passo 2: Configurar Variáveis de Ambiente

Configure as variáveis de ambiente para o app e o modelo:

```bash
# No diretório app/
cp .env.example .env
# Edite .env com as credenciais e configurações AWS

# No diretório maternal-health-risk/
cp .env.example .env
# Edite .env com as mesmas configurações
```

Variáveis necessárias:
```env
AWS_ACCESS_KEY_ID=seu_access_key
AWS_SECRET_ACCESS_KEY=seu_secret_key
AWS_REGION=us-east-1
AWS_ROLE_SAGEMAKER=arn:aws:iam::ACCOUNT:role/maternal-health-system-sagemaker-role
AWS_SAGEMAKER_BUCKET=seu-bucket-sagemaker
AWS_S3_AUDIO_BUCKET=seu-bucket-audio
AWS_TRANSCRIBE_ROLE=arn:aws:iam::ACCOUNT:role/TranscribeDataAccess
LITELLM_API_KEY=sua_api_key  # ou OPENAI_API_KEY
```

### Passo 3: Treinar e Fazer Deploy do Modelo

```bash
cd maternal-health-risk
pip install -r requirements.txt
python deploy.py
```

Aguarde o treinamento e deploy completarem (~10-15 minutos).

### Passo 4: Executar a Aplicação

```bash
cd app
pip install -r requirements.txt
python app.py
```

A aplicação estará disponível em `http://localhost:7860`

### (Opcional) Passo 5: Gerar PDFs de Teste

Para testar o pré-preenchimento automático via PDF, gere laudos de exemplo na raiz do projeto:

```bash
# Instale a dependência (necessário apenas uma vez)
pip install fpdf2

# Gerar os três tipos de laudo de uma vez
python generate_pdf.py

# Ou gerar casos específicos
python generate_pdf.py --low     # somente baixo risco
python generate_pdf.py --high    # somente alto risco
python generate_pdf.py --random  # somente aleatório
```

Arquivos gerados na raiz do projeto:

| Arquivo | Caso | Idade | PA | Glicemia | Temp |
|---|---|---|---|---|---|
| `laudo_baixo_risco.pdf` | Baixo Risco | 25 anos | 110x70 mmHg | 117 mg/dL (6.5 mmol/L) | 36.7°C |
| `laudo_alto_risco.pdf` | Alto Risco | 40 anos | 150x100 mmHg | 189 mg/dL (10.5 mmol/L) | 37.5°C |
| `laudo_medico_exemplo.pdf` | Aleatório | variável | variável | variável | variável |

> **Conversão automática**: o PDF usa unidades clínicas brasileiras (mg/dL, °C). O sistema de pré-preenchimento converte automaticamente para as unidades do modelo (mmol/L, °F) ao processar o PDF.

## 📁 Estrutura do Projeto

```
pos-tech-challenge-04-new/
├── app/                          # Aplicação principal
│   ├── agents/                   # Agentes CrewAI
│   │   └── task_templates.py     # Prompts das tasks (separados da lógica)
│   ├── config/                   # Configurações e constantes
│   ├── models/                   # Modelos de dados (Pydantic)
│   ├── services/                 # Serviços AWS (S3, SageMaker, Transcribe...)
│   ├── tools/                    # Ferramentas dos agentes CrewAI
│   ├── ui/                       # Interface Gradio + handlers de tempo real
│   ├── utils/                    # Utilitários compartilhados (parse_s3_path...)
│   ├── app.py                    # Ponto de entrada
│   └── README.md                 # Documentação do app
│
├── infra/                        # Infraestrutura Terraform
│   ├── main.tf                   # Recursos principais
│   ├── variables.tf              # Variáveis
│   ├── outputs.tf                # Outputs
│   ├── terraform.tfvars.example  # Exemplo de configuração
│   └── README.md                 # Documentação da infra
│
├── maternal-health-risk/         # Modelo de ML
│   ├── code/                     # Código do modelo
│   │   ├── train.py              # Script de treinamento
│   │   ├── inference.py          # Script de inferência
│   │   └── maternal_health_risk.csv  # Dataset
│   ├── deploy.py                 # Script de deploy
│   ├── demo.py                   # Script de demonstração (casos de teste)
│   ├── requirements.txt          # Dependências
│   └── README.md                 # Documentação do modelo
│
├── generate_pdf.py               # Gerador de laudos PDF para testes
├── laudo_baixo_risco.pdf         # Laudo de referência — Baixo Risco (gerado)
├── laudo_alto_risco.pdf          # Laudo de referência — Alto Risco (gerado)
├── laudo_medico_exemplo.pdf      # Laudo aleatório de exemplo (gerado)
└── README.md                     # Este arquivo
```

## 🔄 Fluxo de Dados

1. **Entrada do Usuário**: Dados biométricos, áudio e/ou sinais cardíacos via interface Gradio
2. **Processamento Paralelo**:
   - Dados biométricos → SageMaker Endpoint → Predição de risco
   - Áudio → AWS Transcribe → Transcrição → Análise emocional
   - Sinais cardíacos → Análise de PCG → Detecção de anomalias
3. **Orquestração**: Agentes CrewAI coordenam as análises
4. **Consolidação**: Obstetra Sênior gera relatório final em português
5. **Saída**: Relatório completo com recomendações

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python 3.8+**: Linguagem principal
- **CrewAI**: Framework de agentes de IA
- **Gradio**: Interface web interativa
- **LiteLLM**: Proxy para modelos de linguagem

### Machine Learning
- **XGBoost**: Modelo de classificação
- **AWS SageMaker**: Treinamento e deploy de modelos
- **scikit-learn**: Pré-processamento e métricas

### AWS Services
- **SageMaker**: ML training e inference
- **Transcribe**: Transcrição de áudio
- **Comprehend Medical**: Análise de entidades médicas
- **Textract**: Extração de texto de PDFs
- **S3**: Armazenamento de arquivos
- **CloudWatch**: Logs e monitoramento

### Infraestrutura
- **Terraform**: Infrastructure as Code
- **IAM**: Gerenciamento de permissões
- **S3**: Armazenamento de dados

## 📊 Funcionalidades

### Análise Biométrica
- Predição de risco baseada em sinais vitais
- Classificação binária (Alto Risco / Baixo Risco)
- Probabilidade de risco calculada

### Análise de Áudio
- Transcrição automática de consultas
- Análise emocional e psicológica
- Detecção de sinais de ansiedade ou depressão

### Análise de Sinais Cardíacos
- Processamento de fonocardiogramas (PCG)
- Detecção de anomalias na frequência cardíaca materna
- Análise de variabilidade

### Relatório Integrado
- Consolidação de todas as análises
- Relatório em português brasileiro
- Recomendações baseadas em evidências

## 🔒 Segurança

- Criptografia AES256 em todos os buckets S3
- Acesso público bloqueado
- Políticas IAM com princípio de menor privilégio
- Logs centralizados no CloudWatch
- Versionamento de dados

## 💰 Custos Estimados

### Infraestrutura Base
- **S3 Storage**: ~$0.023/GB/mês
- **CloudWatch Logs**: ~$0.50/GB

### SageMaker
- **Training Job** (ml.m5.large): ~$0.115/hora (~$0.02 por treinamento)
- **Serverless Endpoint**: Pay-per-use (~$0.000004/ms)

### Transcribe
- **Transcription**: ~$0.024/minuto de áudio

**Dica**: Delete endpoints e recursos não utilizados para evitar custos.

## 🐛 Troubleshooting

### Problemas Comuns

1. **Erro de Credenciais AWS**
   - Verifique se as variáveis de ambiente estão configuradas
   - Confirme que as credenciais têm as permissões necessárias

2. **Endpoint SageMaker não encontrado**
   - Verifique se o modelo foi deployado corretamente
   - Confirme o nome do endpoint nas variáveis de ambiente

3. **Erro de Transcrição**
   - Verifique se o bucket S3 está configurado
   - Confirme permissões da role do Transcribe

4. **Erro de Bucket S3**
   - Os nomes de buckets devem ser únicos globalmente
   - Verifique se foram criados pelo Terraform

Para mais detalhes, consulte os READMEs específicos de cada módulo.

## 📚 Documentação Adicional

- [Documentação do App](app/README.md)
- [Documentação da Infraestrutura](infra/README.md)
- [Documentação do Modelo de ML](maternal-health-risk/README.md)

## 🔗 Links Úteis

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

## 📝 Sobre o Projeto

Este projeto foi desenvolvido como parte do trabalho de pós-graduação em tecnologia, focando na aplicação de inteligência artificial e serviços de nuvem para análise de saúde materna.

---

**📌 Nota**: Este é um projeto acadêmico desenvolvido para fins educacionais e de pesquisa. Para uso em ambiente de produção, seriam necessárias revisões adicionais de segurança, testes mais abrangentes, monitoramento adequado e conformidade com regulamentações de saúde.