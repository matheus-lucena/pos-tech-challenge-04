# Violence Against Women — BERT Classifier

BERTimbau (`neuralmind/bert-base-portuguese-cased`) fine-tunado para detecção de violência contra mulher em texto em português.

---

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | **0.9599** |
| F1 macro | **0.9525** |
| F1 violence | **0.9337** |

---

## Estrutura

```
violence-against-women-bert/
├── code/
│   ├── train.py             # Fine-tuning
│   └── inference.py         # Inferência local + FineTunedViolenceDetector
├── data/
│   ├── violence_dataset_train.json
│   ├── violence_dataset_val.json
│   ├── violence_dataset_test.json
│   └── violence_dataset.csv
├── model_output/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── special_tokens_map.json
│   └── metadata.json        # Métricas e hiperparâmetros do treino
├── generate_dataset.py      # Geração de dataset via AWS Bedrock
└── .env.example
```

---

## Dataset

Gerado via AWS Bedrock (`amazon.nova-micro-v1:0`) com 50 cenários:

- **25 cenários de violência** — ameaças de morte, agressão física, violência doméstica, assédio, stalking, revenge porn, cárcere privado, entre outros.
- **25 cenários seguros** — rotina, trabalho, saúde, lazer, relacionamentos saudáveis.
- **Contextos ambíguos** — linguagem intensa sem violência real (jogos online, esportes de combate, culinária, expressões populares) para reduzir falsos positivos.

Split: **80% treino / 10% validação / 10% teste**.

### Gerar novo dataset

```bash
cp .env.example .env
# preencher AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY

# Gerar e dividir (80 frases por cenário)
python generate_dataset.py --split

# Modelo alternativo
python generate_dataset.py --model amazon.nova-lite-v1:0 --split

# Adicionar apenas cenários novos ao dataset existente
python generate_dataset.py --append --split
```

---

## Treinamento

```bash
python code/train.py \
    --train data/violence_dataset_train.json \
    --val   data/violence_dataset_val.json \
    --model-dir model_output
```

### Parâmetros

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--epochs` | `4` | Épocas |
| `--batch-size` | `16` | Tamanho do batch |
| `--learning-rate` | `2e-5` | AdamW learning rate |
| `--warmup-ratio` | `0.1` | Proporção de warmup steps |
| `--max-length` | `128` | Tokens máximos por entrada |
| `--patience` | `2` | Early stopping por `f1_violence` |

Em CPU ou com poucos dados:

```bash
python code/train.py --epochs 2 --batch-size 8
```

O checkpoint com melhor `f1_violence` na validação é salvo em `model_output/`.

---

## Inferência

### Linha de comando

```bash
python code/inference.py --model-dir model_output --text "Vou te matar se você sair de casa."
```

```
Texto     : Vou te matar se você sair de casa.
Violência : True
Categoria : violence
Score     : 0.9821
```

### Código Python

```python
import sys
sys.path.insert(0, "violence-against-women-bert")
from code.inference import FineTunedViolenceDetector

detector = FineTunedViolenceDetector("violence-against-women-bert/model_output", threshold=0.5)
is_violent, category, score = detector.predict("Você não vai a lugar nenhum.")
```

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `VIOLENCE_THRESHOLD` | `0.5` | Limiar para classificar como violência |
| `VIOLENCE_BERT_DEVICE` | auto | Forçar `cpu` ou `cuda` |

---

## Integração com o app

`FineTunedViolenceDetector` expõe a mesma interface de `ZeroShotViolenceDetector` usada em `app/services/transcribe_streaming_service.py`:

```python
predict(text: str) -> tuple[bool, str, float]
# retorna: (is_violent, category, score)
```

---

## Dependências

```bash
pip install -r requirements.txt
```

Ou instale diretamente:

```bash
pip install torch>=2.1.0 transformers>=4.46.0 scikit-learn>=1.3.0 \
            pandas>=2.0.0 numpy>=1.24.0 python-dotenv>=1.0.0 boto3>=1.34.0
```

---

## Modelo base

[`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased) — BERTimbau, pré-treinado em português do Brasil.
