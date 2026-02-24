# 🏥 Maternal Health Analysis System - App

Sistema multimodal de análise de saúde materna que utiliza inteligência artificial para avaliar dados biométricos, áudios de consulta e sinais cardíacos maternos, fornecendo uma avaliação completa do risco de saúde materna.

## 📋 Descrição

Este aplicativo é uma interface web construída com **Gradio** que integra múltiplos serviços AWS e agentes CrewAI para análise de saúde materna. O sistema processa:

- **Dados Biométricos**: Idade, pressão arterial, glicemia, temperatura corporal e frequência cardíaca
- **Áudios de Consulta**: Transcrição e análise emocional usando AWS Transcribe
- **Sinais Cardíacos Maternos**: Análise de fonocardiograma (PCG) para detecção de anomalias

## 🏗️ Arquitetura

O sistema utiliza uma arquitetura baseada em agentes CrewAI:

- **Biometric Analyst**: Analisa sinais vitais via SageMaker
- **Perinatal Psychologist**: Detecta sinais de ansiedade ou depressão em áudios
- **Maternal Monitoring Specialist**: Analisa sinais cardíacos maternos (PCG)
- **Obstetra Sênior**: Consolida todas as análises em um relatório final em português

## 🚀 Tecnologias

- **CrewAI**: Framework de agentes de IA
- **Gradio**: Interface web interativa
- **AWS Services**:
  - SageMaker: Modelos de ML para predição de risco
  - Transcribe: Transcrição de áudio
  - Comprehend Medical: Análise de entidades médicas
  - Textract: Extração de texto de PDFs
  - S3: Armazenamento de arquivos
- **LiteLLM**: Proxy para modelos de linguagem
- **Python**: Linguagem principal

## 📦 Instalação

### Pré-requisitos

- Python 3.8+
- Credenciais AWS configuradas
- Variáveis de ambiente configuradas (veja `.env.example`)

### Passos

1. Clone o repositório e navegue até a pasta `app`:
```bash
cd app
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas credenciais AWS e configurações
```

## ⚙️ Configuração

### Variáveis de Ambiente Necessárias

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# SageMaker
AWS_ROLE_SAGEMAKER=arn:aws:iam::account:role/sagemaker-role
AWS_SAGEMAKER_BUCKET=your-sagemaker-bucket
AWS_SAGEMAKER_ENDPOINT=your-endpoint-name

# S3
AWS_S3_AUDIO_BUCKET=your-audio-bucket

# Transcribe
AWS_TRANSCRIBE_ROLE=arn:aws:iam::account:role/TranscribeDataAccess

# LiteLLM / LLM
LITELLM_API_KEY=your_api_key
# ou
OPENAI_API_KEY=your_openai_key
```

## 🎯 Uso

### Executar a Aplicação

```bash
python app.py
```

A aplicação estará disponível em `http://localhost:7860`

### Funcionalidades da Interface

1. **Análise Biométrica**: 
   - Insira dados vitais (idade, pressão arterial, glicemia, temperatura, frequência cardíaca)
   - O sistema consulta o modelo SageMaker para predição de risco

2. **Análise de Áudio**:
   - Faça upload de um arquivo de áudio de consulta
   - O sistema transcreve e analisa o conteúdo emocional

3. **Análise de Sinais Cardíacos**:
   - Faça upload de arquivo de áudio com sinais cardíacos maternos
   - O sistema analisa anomalias na frequência cardíaca materna

4. **Análise Completa**:
   - Combine todas as análises para um relatório completo
   - O agente Obstetra Sênior consolida todas as informações

## 📁 Estrutura do Projeto

```
app/
├── agents/              # Agentes CrewAI
│   └── crew_orchestrator.py
├── config/              # Configurações
│   └── llm_config.py
├── models/              # Modelos de dados
│   └── report.py
├── services/            # Serviços AWS
│   ├── s3_service.py
│   ├── sagemaker_service.py
│   ├── transcribe_service.py
│   ├── comprehend_medical_service.py
│   └── ...
├── tools/               # Ferramentas dos agentes
│   ├── health_tools.py
│   └── maternal_tools.py
├── ui/                  # Interface Gradio
│   ├── gradio_interface.py
│   ├── processors.py
│   └── realtime_processor.py
├── app.py              # Ponto de entrada
└── requirements.txt    # Dependências
```

## 🔧 Desenvolvimento

### Adicionar Novos Agentes

1. Crie uma nova ferramenta em `tools/`
2. Adicione o agente em `agents/crew_orchestrator.py`
3. Crie uma task correspondente

### Adicionar Novos Serviços AWS

1. Crie um novo serviço em `services/`
2. Implemente métodos para interagir com o serviço AWS
3. Integre com os agentes ou interface

## 🐛 Troubleshooting

### Erro de Credenciais AWS
- Verifique se as variáveis de ambiente estão configuradas
- Confirme que as credenciais têm as permissões necessárias

### Erro de Endpoint SageMaker
- Verifique se o endpoint está ativo
- Confirme o nome do endpoint nas variáveis de ambiente

### Erro de Transcrição
- Verifique se o bucket S3 está configurado corretamente
- Confirme que a role do Transcribe tem permissões adequadas

## 🎤 Transcrição em Tempo Real e Detecção de Violência

O sistema inclui um módulo de processamento de áudio em tempo real via **AWS Transcribe Streaming**:

### Fluxo
1. O microfone captura áudio via `PyAudio` e divide em chunks de `~100ms`
2. Os chunks são enviados para o `TranscribeStreamingService` que mantém uma conexão WebSocket com o AWS Transcribe
3. Os resultados (parciais e finais) são entregues ao `RealtimeAudioProcessor` através de uma fila thread-safe
4. A interface Gradio exibe a transcrição em tempo real com polling a cada `200ms`

### Detecção de Violência
- Implementada via **zero-shot classification** com o modelo `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Analisa janelas de contexto (`CONTEXT_WINDOW_SIZE = 5` segmentos finais)
- Combina lista de palavras-chave de perigo com classificação probabilística
- Emite alertas visuais na interface quando `score > VIOLENCE_THRESHOLD (0.75)`

### Dependências extras para streaming
```bash
pip install amazon-transcribe transformers torch pyaudio
```

> **Nota**: `torch` é necessário para o modelo de detecção de violência. A instalação sem GPU funciona
> normalmente em CPU, mas pode ser mais lenta. Passe `use_cuda=True` ao instanciar `ZeroShotViolenceDetector`
> para habilitar GPU se disponível.

### Módulos relacionados

| Módulo | Responsabilidade |
|---|---|
| `services/transcribe_streaming_service.py` | Conexão WebSocket com AWS Transcribe, detecção de violência |
| `ui/realtime_processor.py` | Captura de microfone, gerenciamento de threads, gravação WAV |
| `ui/realtime_handlers.py` | Handlers Gradio para start/stop/update da transcrição em tempo real |

## 📝 Sobre o Projeto

Este projeto faz parte do sistema de saúde materna desenvolvido para o trabalho de pós-graduação.

