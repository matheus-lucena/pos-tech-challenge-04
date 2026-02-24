"""
Gerador de dataset via AWS Bedrock para detecção de violência contra mulher.

Saída JSON:
  [{"text": "...", "label": 1, "category": "ameaça de morte"}, ...]

Uso:
    python generate_dataset.py --split
    python generate_dataset.py --model amazon.nova-lite-v1:0 --split
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL  = "amazon.nova-micro-v1:0"
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
BATCH_SIZE     = 40
MAX_RETRIES    = 5

# ---------------------------------------------------------------------------
# Cenários: (label, category)
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[int, str]] = [
    # --- VIOLÊNCIA (label=1) ---
    (1, "ameaça de morte ou lesão corporal grave"),
    (1, "agressão física: espancamento, murro, chute, empurrão"),
    (1, "violência doméstica e controle coercitivo"),
    (1, "assédio verbal: xingamentos e humilhação"),
    (1, "manipulação emocional e gaslighting"),
    (1, "perseguição e stalking presencial"),
    (1, "stalking digital: monitoramento de redes e mensagens"),
    (1, "ameaça de tirar os filhos como coação"),
    (1, "chantagem e ameaça de expor fotos íntimas"),
    (1, "abuso sexual dentro do relacionamento"),
    (1, "estupro ou tentativa de estupro"),
    (1, "revenge porn: divulgação de conteúdo íntimo"),
    (1, "violência financeira: controle de dinheiro"),
    (1, "isolamento social: proibição de ver família e amigos"),
    (1, "ameaça com arma branca ou de fogo"),
    (1, "pedido de socorro em situação de perigo imediato"),
    (1, "ameaça após separação ou término"),
    (1, "violência durante a gravidez"),
    (1, "cárcere privado dentro de casa"),
    (1, "ameaça aos filhos ou familiares para controlar a vítima"),
    (1, "assédio sexual no trabalho"),
    (1, "humilhação pública e difamação"),
    (1, "ameaça velada e linguagem de duplo sentido intimidatória"),
    (1, "violência praticada por familiar ou parente"),
    (1, "pressão para abandonar emprego ou estudos"),
    # --- SEGURO (label=0) ---
    (0, "rotina e atividades cotidianas neutras"),
    (0, "trabalho e carreira de forma positiva"),
    (0, "saúde e bem-estar: consultas, exames, exercícios"),
    (0, "família e relacionamentos saudáveis"),
    (0, "notícias e eventos culturais neutros"),
    (0, "lazer, viagens e hobbies"),
    (0, "educação e aprendizado"),
    (0, "alimentação e culinária"),
    (0, "natureza, clima e meio ambiente"),
    (0, "tecnologia e ciência de forma informativa"),
    (0, "finanças pessoais e planejamento"),
    (0, "esportes e atividade física"),
    (0, "relacionamento amoroso saudável e respeitoso"),
    (0, "amizade e socialização descontraída"),
    (0, "cuidado de pets e animais"),
    (0, "música, filmes, séries e entretenimento"),
    (0, "voluntariado e solidariedade"),
    (0, "empreendedorismo e projetos criativos"),
    (0, "decoração, casa e jardim"),
    (0, "religião e espiritualidade serena"),
    (0, "humor e situações engraçadas do cotidiano"),
    (0, "elogios e reconhecimento entre pessoas"),
    (0, "planejamento de festas e comemorações"),
    (0, "discussão construtiva e resolução pacífica de conflitos"),
    (0, "apoio emocional e palavras de encorajamento"),
    # --- CONTEXTOS AMBÍGUOS — linguagem intensa mas sem violência real (label=0) ---

    # Objetos cortantes em uso doméstico/culinário
    (0, "culinária e preparo de refeições usando facas e utensílios de cozinha"),
    (0, "uso cotidiano de ferramentas como tesoura, martelo, serrote em casa"),
    (0, "jardinagem e uso de ferramentas de corte no jardim"),
    (0, "açougue, pesca ou caça esportiva descrita de forma neutra"),
    (0, "bricolagem, marcenaria e trabalhos manuais em casa"),

    # Jogos online e videogames — ameaças e xingamentos no contexto de partida
    (0, "jogos online multiplayer com trash talk e provocações entre jogadores sem violência real"),
    (0, "partida de FPS ou battle royale onde jogadores falam 'vou te matar', 'te destruo', 'morreu' no jogo"),
    (0, "streamer ou gamer comentando jogadas agressivas, mortes e eliminações no game"),
    (0, "jogo de luta ou RPG com falas de personagens ameaçando inimigos fictícios"),
    (0, "torcida ou comemoração exaltada em jogo online, com gritos e palavrões sem ameaça real"),

    # Esportes de combate e artes marciais
    (0, "treino de artes marciais, boxe ou MMA descrito de forma técnica e esportiva"),
    (0, "comentarista ou lutador descrevendo golpes, nocautes e combates em luta esportiva"),
    (0, "atleta relatando lesão ou momento de dor durante treino ou competição esportiva"),

    # Entretenimento com temática de violência fictícia
    (0, "discussão sobre cena de ação, crime ou thriller em filme, série ou livro"),
    (0, "comentário sobre true crime, documentário policial ou podcast de crimes reais"),
    (0, "pessoas discutindo enredo de novela ou série com personagens em situação de perigo fictício"),
    (0, "entusiasta de história descrevendo batalhas, guerras ou conflitos históricos"),

    # Linguagem figurada e expressões populares
    (0, "uso de expressões populares hiperbólicas como 'me mato de rir', 'quase morri de susto', 'tô morta de cansaço'"),
    (0, "brincadeiras e zoações entre amigos usando linguagem exagerada sem intenção real de ameaça"),
    (0, "discussão acalorada entre pessoas que termina em entendimento mútuo"),
    (0, "pessoas relatando susto, nervosismo ou medo sem situação de perigo real"),

    # Profissões e contextos técnicos
    (0, "profissional de saúde, médico ou enfermeiro descrevendo procedimento cirúrgico ou acidente"),
    (0, "bombeiro, policial ou socorrista relatando ocorrência de forma técnica e neutra"),
    (0, "jornalista ou repórter noticiando fato violento de forma objetiva e impessoal"),
    (0, "advogado, delegado ou promotor discutindo caso criminal em linguagem jurídica"),

    # Situações cotidianas com linguagem de susto
    (0, "crianças brincando de luta, super-herói ou guerra de mentira sem agressão real"),
    (0, "situações de susto ou preocupação sem violência real, como acidente doméstico leve"),
    (0, "pessoa relatando pesadelo, sonho ruim ou cena assustadora de forma descontraída"),
]

PROMPT = """Gere EXATAMENTE {n} frases em português brasileiro coloquial sobre o cenário:
"{category}"

Regras:
- Frases realistas, como falas reais em áudios, ligações ou conversas do dia a dia
- Varie ponto de vista e estrutura
- Entre 8 e 50 palavras por frase
- SEM numeração
- Para cenários SEGUROS: as frases NÃO devem conter nenhuma ameaça, agressão ou perigo — mesmo que mencionem objetos como faca, tesoura ou ferramentas, o contexto deve ser claramente doméstico/culinário/neutro
- Retorne APENAS um array JSON de strings

["frase 1", "frase 2", ...]"""

# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------

class BedrockGenerator:
    def __init__(self, model_id: str, region: str):
        self.model_id = model_id.replace("us.bedrock/", "")
        self.client   = boto3.client("bedrock-runtime", region_name=region)
        print(f"Modelo: {self.model_id} | Região: {region}\n")

    def _call(self, prompt: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 4096, "temperature": 0.95},
                )
                return resp["output"]["message"]["content"][0]["text"]
            except ClientError as e:
                if e.response["Error"]["Code"] in (
                    "ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"
                ):
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  Throttling — aguardando {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Bedrock indisponível após todas as tentativas.")

    def _parse(self, raw: str) -> list[str]:
        raw = raw.strip()
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e == 0:
            return [l.strip().strip('"-,') for l in raw.splitlines() if len(l.strip()) > 8]
        try:
            return [str(x).strip() for x in json.loads(raw[s:e]) if str(x).strip()]
        except json.JSONDecodeError:
            return [l.strip().strip('"-,') for l in raw[s:e].splitlines() if len(l.strip()) > 8]

    def generate(self, scenarios: list[tuple[int, str]], per_scenario: int) -> list[dict]:
        results: list[dict] = []
        seen: set[str]      = set()

        for i, (label, category) in enumerate(scenarios, 1):
            print(f"  [{i:>2}/{len(scenarios)}] {category}")
            collected = []

            while len(collected) < per_scenario:
                n = min(BATCH_SIZE, per_scenario - len(collected) + 5)
                try:
                    sentences = self._parse(self._call(PROMPT.format(n=n, category=category)))
                except Exception as e:
                    print(f"    Erro: {e} — pulando.")
                    time.sleep(2)
                    break

                for s in sentences:
                    if s not in seen and len(s.split()) >= 3:
                        seen.add(s)
                        collected.append(s)
                        if len(collected) >= per_scenario:
                            break

            for text in collected[:per_scenario]:
                results.append({"text": text, "label": label, "category": category})

        return results

# ---------------------------------------------------------------------------
# Salvar / carregar JSON
# ---------------------------------------------------------------------------

def save_json(samples: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    v = sum(1 for s in samples if s["label"] == 1)
    print(f"  → {path}  ({len(samples)} amostras | {v} violência / {len(samples)-v} seguro)")


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_and_save(samples: list[dict], output: str) -> None:
    random.shuffle(samples)
    n  = len(samples)
    t  = int(n * 0.8)
    v  = t + int(n * 0.1)
    base, stem = Path(output).parent, Path(output).stem
    print("\nSalvando splits:")
    save_json(samples[:t],  str(base / f"{stem}_train.json"))
    save_json(samples[t:v], str(base / f"{stem}_val.json"))
    save_json(samples[v:],  str(base / f"{stem}_test.json"))


def _existing_categories(splits_dir: str, stem: str) -> set[str]:
    """Retorna o conjunto de categorias já presentes nos arquivos existentes."""
    categories: set[str] = set()
    for suffix in ("_train.json", "_val.json", "_test.json", ".json"):
        path = Path(splits_dir) / f"{stem}{suffix}"
        if path.exists():
            for record in load_json(str(path)):
                cat = record.get("category", "")
                if cat:
                    categories.add(cat)
    return categories


def _load_all_splits(splits_dir: str, stem: str) -> list[dict]:
    """Carrega todos os splits existentes em uma lista única."""
    all_records: list[dict] = []
    for suffix in ("_train.json", "_val.json", "_test.json"):
        path = Path(splits_dir) / f"{stem}{suffix}"
        if path.exists():
            all_records.extend(load_json(str(path)))
    if not all_records:
        full = Path(splits_dir) / f"{stem}.json"
        if full.exists():
            all_records = load_json(str(full))
    return all_records

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",        default="data/violence_dataset.json")
    p.add_argument("--per-scenario",  type=int, default=80,
                   help="Frases por cenário (padrão: 80)")
    p.add_argument("--model",         default=DEFAULT_MODEL)
    p.add_argument("--region",        default=DEFAULT_REGION)
    p.add_argument("--split",         action="store_true")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--append",        action="store_true",
                   help="Gera apenas cenários novos e mescla com o dataset existente")
    args = p.parse_args()

    random.seed(args.seed)
    base = Path(args.output).parent
    stem = Path(args.output).stem

    if args.append:
        # Descobre quais categorias já existem
        existing_cats = _existing_categories(str(base), stem)
        new_scenarios  = [(l, c) for l, c in SCENARIOS if c not in existing_cats]

        if not new_scenarios:
            print("Nenhum cenário novo encontrado — dataset já está atualizado.")
            return

        print(f"Cenários existentes : {len(existing_cats)}")
        print(f"Cenários novos      : {len(new_scenarios)}")
        for l, c in new_scenarios:
            print(f"  {'[violência]' if l == 1 else '[seguro]  '} {c}")
        print()

        gen      = BedrockGenerator(args.model, args.region)
        new_data = gen.generate(new_scenarios, per_scenario=args.per_scenario)

        existing = _load_all_splits(str(base), stem)
        all_samples = existing + new_data
        random.shuffle(all_samples)
        print(f"\nMesclado: {len(existing)} existentes + {len(new_data)} novos = {len(all_samples)} total")
    else:
        violence_n = sum(1 for l, _ in SCENARIOS if l == 1)
        safe_n     = sum(1 for l, _ in SCENARIOS if l == 0)
        print(f"Cenários : {violence_n} violência + {safe_n} seguro = {len(SCENARIOS)} total")
        print(f"Frases   : {args.per_scenario} por cenário → ~{len(SCENARIOS) * args.per_scenario} amostras\n")

        gen         = BedrockGenerator(args.model, args.region)
        all_samples = gen.generate(SCENARIOS, per_scenario=args.per_scenario)
        random.shuffle(all_samples)

    if args.split:
        split_and_save(all_samples, args.output)
    else:
        print("\nSalvando dataset:")
        save_json(all_samples, args.output)


if __name__ == "__main__":
    main()
