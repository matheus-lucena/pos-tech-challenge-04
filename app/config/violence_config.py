"""
Configurações do detector de violência contra mulher.

Centraliza todos os labels, hipóteses e keywords para que o
transcribe_streaming_service.py não precise defini-los internamente.
"""

# ---------------------------------------------------------------------------
# Zero-shot: estágio binário (Stage 1)
# ---------------------------------------------------------------------------

BINARY_DANGER = "situação de risco, perigo ou violência"
BINARY_SAFE   = "situação normal e segura"
BINARY_HYPOTHESIS = "Esta frase descreve uma {}."
BINARY_THRESHOLD  = 0.55

# ---------------------------------------------------------------------------
# Zero-shot: categorias de risco (Stage 2)
# ---------------------------------------------------------------------------

CATEGORY_LABELS = [
    "violência física ou ameaça de morte",
    "perseguição ou stalking",
    "assédio verbal, psicológico ou online",
    "posse ou uso de arma",
    "exploração ou abuso sexual de menor",
    "violência doméstica",
    "ameaça velada ou coação",
    "pedido de socorro ou emergência",
    "exposição não consentida de conteúdo íntimo",
]

CATEGORY_HYPOTHESIS = "Esta frase é sobre {}."
