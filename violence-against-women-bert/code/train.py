"""
Fine-tuning de BERTimbau para detecção de violência contra mulher.

Modelo base : neuralmind/bert-base-portuguese-cased
Tarefa      : Classificação binária — 0 (seguro) / 1 (violência)

Uso:
    python code/train.py \
        --train data/violence_dataset_train.json \
        --val   data/violence_dataset_val.json \
        --model-dir model_output

    # CPU / poucos dados:
    python code/train.py ... --epochs 2 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import BertForSequenceClassification, BertTokenizerFast, get_linear_schedule_with_warmup

BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
ID2LABEL   = {0: "safe", 1: "violence"}
LABEL2ID   = {"safe": 0, "violence": 1}
MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ViolenceDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = MAX_LENGTH):
        texts  = [r["text"] for r in records]
        labels = [int(r["label"]) for r in records]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ---------------------------------------------------------------------------
# Carregamento de dados — aceita JSON ou CSV, arquivo ou diretório
# ---------------------------------------------------------------------------

def load_records(path: str, stem: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        for ext in (".json", ".csv"):
            f = p / f"{stem}{ext}"
            if f.exists():
                p = f
                break
        else:
            candidates = list(p.glob("*.json")) + list(p.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError(f"Nenhum .json ou .csv em {path}")
            p = candidates[0]
    print(f"  Carregando: {p}")
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    df = pd.read_csv(p)
    return df[["text", "label"]].dropna().to_dict("records")

# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits
            preds  = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    return {
        "accuracy":    round(accuracy_score(all_labels, all_preds), 4),
        "f1_macro":    round(f1_score(all_labels, all_preds, average="macro"), 4),
        "f1_violence": round(f1_score(all_labels, all_preds, pos_label=1, average="binary"), 4),
    }

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Modelo base : {args.base_model}")
    print(f"Épocas      : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.learning_rate}\n")

    # 1. Dados
    print("[1/4] Carregando dados...")
    train_records = load_records(args.train, args.train_file)
    val_records   = load_records(args.val,   args.val_file)
    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}")

    # 2. Tokenizador e datasets
    print("\n[2/4] Tokenizando...")
    tokenizer     = BertTokenizerFast.from_pretrained(args.base_model)
    train_dataset = ViolenceDataset(train_records, tokenizer, args.max_length)
    val_dataset   = ViolenceDataset(val_records,   tokenizer, args.max_length)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size * 2)

    # 3. Modelo
    print("\n[3/4] Carregando modelo base...")
    model = BertForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    # 4. Loop de treino
    print("\n[4/4] Treinando...\n")
    model_dir    = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_f1      = 0.0
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            loss   = model(**inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(f"  Época {epoch}/{args.epochs} | step {step:>4}/{len(train_loader)} | loss: {total_loss/step:.4f}")

        metrics = evaluate(model, val_loader, device)
        print(f"  → Val | acc: {metrics['accuracy']}  f1_macro: {metrics['f1_macro']}  f1_violence: {metrics['f1_violence']}\n")

        # Salva melhor modelo (early stopping por f1_violence)
        if metrics["f1_violence"] > best_f1:
            best_f1 = metrics["f1_violence"]
            patience_cnt = 0
            model.save_pretrained(str(model_dir))
            tokenizer.save_pretrained(str(model_dir))
            print(f"  Modelo salvo (melhor f1_violence: {best_f1:.4f})\n")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stopping após {args.patience} épocas sem melhora.")
                break

    # Avaliação final + relatório
    print("─" * 50)
    print("Avaliação final no conjunto de validação:")
    final = evaluate(model, val_loader, device)

    # carrega o melhor modelo salvo para o relatório completo
    model_best = BertForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    all_preds, all_labels = [], []
    model_best.eval()
    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            preds  = torch.argmax(model_best(**inputs).logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    print(classification_report(all_labels, all_preds, target_names=["safe", "violence"]))

    # Metadados
    metadata = {
        "base_model": args.base_model,
        "max_length": args.max_length,
        "id2label":   ID2LABEL,
        "label2id":   LABEL2ID,
        "best_f1_violence": best_f1,
        "final_metrics":    final,
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nModelo final salvo em : {model_dir}")
    print(f"Melhor f1_violence    : {best_f1:.4f}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train",          default="../data")
    p.add_argument("--val",            default="../data")
    p.add_argument("--model-dir",      default="../model_output")
    p.add_argument("--train-file",     default="violence_dataset_train")
    p.add_argument("--val-file",       default="violence_dataset_val")
    p.add_argument("--base-model",     default=BASE_MODEL)
    p.add_argument("--epochs",         type=int,   default=4)
    p.add_argument("--batch-size",     type=int,   default=16)
    p.add_argument("--learning-rate",  type=float, default=2e-5)
    p.add_argument("--warmup-ratio",   type=float, default=0.1)
    p.add_argument("--weight-decay",   type=float, default=0.01)
    p.add_argument("--max-length",     type=int,   default=MAX_LENGTH)
    p.add_argument("--patience",       type=int,   default=2)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
