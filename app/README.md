# App — Maternal Health Analysis

Interface web e orquestração de agentes para análise multimodal de saúde materna.

## Componentes

- **Interface Gradio** — entrada de dados biométricos, upload de áudio e sinais cardíacos
- **Agentes CrewAI** — quatro agentes especializados coordenados em crew:
  - **Biometric Analyst** — analisa sinais vitais via SageMaker
  - **Perinatal Psychologist** — detecta sinais de ansiedade ou depressão no áudio
  - **Maternal Monitoring Specialist** — analisa fonocardiograma (PCG)
  - **Obstetra Sênior** — consolida as análises e gera relatório final em português
- **Transcrição em tempo real** — AWS Transcribe Streaming + detecção de violência contra mulher

---

## Instalação

```bash
cd app
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
cp .env.example .env
# preencher .env
```

---

## Variáveis de Ambiente

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

AWS_ROLE_SAGEMAKER=arn:aws:iam::account:role/sagemaker-role
AWS_SAGEMAKER_BUCKET=
AWS_SAGEMAKER_ENDPOINT=

AWS_S3_AUDIO_BUCKET=
AWS_TRANSCRIBE_ROLE=arn:aws:iam::account:role/TranscribeDataAccess

LITELLM_API_KEY=
# ou
OPENAI_API_KEY=
```

---

## Execução

```bash
python app.py
```

Acesse em `http://localhost:7860`.

---

## Estrutura

```
app/
├── agents/
│   └── crew_orchestrator.py      # Definição dos agentes e tasks
│   └── task_templates.py         # Prompts das tasks
├── config/                       # Configurações e constantes
├── models/                       # Modelos de dados Pydantic
├── services/                     # Serviços AWS
│   ├── s3_service.py
│   ├── sagemaker_service.py
│   ├── transcribe_service.py
│   ├── transcribe_streaming_service.py
│   └── comprehend_medical_service.py
├── tools/                        # Ferramentas dos agentes
├── ui/
│   ├── gradio_interface.py
│   ├── realtime_processor.py     # Captura de microfone, threads, gravação WAV
│   └── realtime_handlers.py      # Handlers Gradio para transcrição em tempo real
├── utils/
└── app.py
```

---

## Transcrição em Tempo Real

Fluxo de captura e análise de áudio ao vivo:

1. `PyAudio` captura o microfone em chunks de ~100ms
2. Os chunks são enviados ao `TranscribeStreamingService` via WebSocket para o AWS Transcribe
3. Resultados parciais e finais chegam ao `RealtimeAudioProcessor` por fila thread-safe
4. A interface Gradio exibe a transcrição com polling a cada 200ms

### Detecção de violência

O módulo de detecção analisa janelas de contexto (`CONTEXT_WINDOW_SIZE = 5` segmentos finais). Dois modos disponíveis:

- **Zero-shot** — modelo `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` com threshold `0.75`
- **Fine-tuned** — `FineTunedViolenceDetector` (BERTimbau, ver `violence-against-women-bert/`)

Alertas visuais são emitidos na interface quando o score ultrapassa o threshold configurado.

| Módulo | Responsabilidade |
|---|---|
| `services/transcribe_streaming_service.py` | Conexão WebSocket com AWS Transcribe, detecção de violência |
| `ui/realtime_processor.py` | Captura de microfone, threads, gravação WAV |
| `ui/realtime_handlers.py` | Handlers Gradio para start/stop/update |

---

## Troubleshooting

**Credenciais AWS inválidas** — verifique as variáveis de ambiente e as permissões da conta.

**Endpoint SageMaker não encontrado** — confirme que o endpoint está `InService` e o nome está correto no `.env`.

**Erro de transcrição** — verifique se o bucket S3 existe e se a role do Transcribe tem acesso a ele.
