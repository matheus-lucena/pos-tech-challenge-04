"""
Inferência local do modelo BERT fine-tunado para detecção de violência contra mulher.

Uso:
    python code/inference.py --model-dir model_output --text "Vou te matar."

Integração com transcribe_streaming_service.py:
    O serviço instancia FineTunedViolenceDetector diretamente.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

DEFAULT_MAX_LENGTH = 128
DEFAULT_ID2LABEL   = {0: "safe", 1: "violence"}
DEFAULT_THRESHOLD  = 0.5


def _resolve_device() -> torch.device:
    """
    Seleciona CPU ou CUDA. Valida os kernels antes de usar a GPU para evitar
    o erro 'no kernel image is available' em drivers incompatíveis.
    Pode ser forçado via VIOLENCE_BERT_DEVICE=cpu|cuda.
    """
    forced = os.environ.get("VIOLENCE_BERT_DEVICE", "").lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda":
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = torch.zeros(1).cuda() + 1
        return torch.device("cuda")
    except RuntimeError:
        print("⚠️  CUDA detectada mas kernels incompatíveis — usando CPU.")
        return torch.device("cpu")


class FineTunedViolenceDetector:
    """
    Detector de violência contra mulher baseado em BERT fine-tunado.

    Interface idêntica ao ZeroShotViolenceDetector do transcribe_streaming_service.py:
        predict(text) → (is_violent: bool, category: str, score: float)

    Args:
        model_dir : Caminho para o diretório gerado por train.py (contém config.json).
        threshold : Limiar de confiança para classificar como violência.
                    Padrão: 0.5, ajustável via VIOLENCE_THRESHOLD.
    """

    def __init__(self, model_dir: str, threshold: float | None = None):
        model_dir = Path(model_dir)

        self.tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
        self.model     = BertForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

        self.device = _resolve_device()
        self.model.to(self.device)

        metadata: dict = {}
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        self.max_length = metadata.get("max_length", DEFAULT_MAX_LENGTH)
        self.id2label   = {int(k): v for k, v in metadata.get("id2label", DEFAULT_ID2LABEL).items()}
        self.threshold  = threshold if threshold is not None else float(
            os.environ.get("VIOLENCE_THRESHOLD", DEFAULT_THRESHOLD)
        )

        print(f"Modelo carregado de : {model_dir}")
        print(f"Device              : {self.device}")
        print(f"Threshold           : {self.threshold}")

    def predict(self, text: str) -> tuple[bool, str, float]:
        """
        Classifica o texto.

        Retorna:
            is_violent (bool)  — True se violência detectada
            category   (str)   — rótulo do modelo ou 'safe'
            score      (float) — probabilidade da classe violência (0-1)
        """
        if not text or len(text.strip()) < 10:
            return False, "too_short", 0.0

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        try:
            inputs_dev = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                probas = torch.softmax(self.model(**inputs_dev).logits, dim=-1).cpu()
        except RuntimeError:
            print("⚠️  Inferência CUDA falhou — retentando em CPU.")
            self.model.to("cpu")
            self.device = torch.device("cpu")
            inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                probas = torch.softmax(self.model(**inputs_cpu).logits, dim=-1).cpu()

        score      = probas[0, 1].item()
        is_violent = score >= self.threshold
        label      = self.id2label.get(1 if is_violent else 0, "safe")
        return is_violent, label, round(score, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste rápido do modelo fine-tunado.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text",      required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    detector = FineTunedViolenceDetector(args.model_dir, threshold=args.threshold)
    is_violent, category, score = detector.predict(args.text)
    print(f"\nTexto     : {args.text}")
    print(f"Violência : {is_violent}")
    print(f"Categoria : {category}")
    print(f"Score     : {score:.4f}")
