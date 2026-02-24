"""
Gerador de laudos médicos em PDF para testes do sistema de análise de saúde materna.

Uso:
    python generate_pdf.py              # gera baixo risco + alto risco + aleatório
    python generate_pdf.py --low        # somente baixo risco
    python generate_pdf.py --high       # somente alto risco
    python generate_pdf.py --random     # somente aleatório

Unidades: parâmetros recebidos em mmol/L e °F (formato do modelo);
o PDF exibe os valores convertidos para mg/dL e °C.
"""

import argparse
import random
from datetime import date, timedelta
from typing import Optional

from fpdf import FPDF, XPos, YPos

_NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}
_CONT = {"new_x": XPos.RIGHT, "new_y": YPos.TOP}

LOW_RISK_CASE = {
    "Age": 25,
    "SystolicBP": 110,
    "DiastolicBP": 70,
    "BS_mmol": 6.5,
    "BodyTemp_f": 98.0,
    "HeartRate": 70,
    "description": "Mulher jovem, pressão normal, glicemia controlada",
}

HIGH_RISK_CASE = {
    "Age": 40,
    "SystolicBP": 150,
    "DiastolicBP": 100,
    "BS_mmol": 10.5,
    "BodyTemp_f": 99.5,
    "HeartRate": 95,
    "description": "Hipertensão arterial, glicemia elevada, múltiplos fatores de risco",
}


def _f_to_c(f: float) -> float:
    return round((f - 32) * 5 / 9, 1)


def _mmol_to_mgdl(mmol: float) -> int:
    return round(mmol * 18.018)


def _random_female_name() -> str:
    first_names = [
        "MARIA", "ANA", "JULIANA", "BEATRIZ", "FERNANDA",
        "CAMILA", "PATRICIA", "LUCIANA", "CAROLINA", "GABRIELA",
    ]
    last_names = [
        "SILVA", "SANTOS", "OLIVEIRA", "SOUZA", "PEREIRA",
        "COSTA", "RODRIGUES", "ALMEIDA", "NASCIMENTO", "LIMA",
    ]
    middle = random.choice(first_names) if random.random() < 0.35 else None
    parts = [random.choice(first_names)]
    if middle:
        parts.append(middle)
    parts.extend([random.choice(last_names), random.choice(last_names)])
    return " ".join(parts)


def _dob_for_age(age: int) -> date:
    today = date.today()
    return today.replace(year=today.year - age) - timedelta(days=random.randint(0, 364))


def _fmt_date_br(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _random_order_id() -> str:
    return f"#{random.randint(100000, 999999)}"


class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "LABORATORIO CENTRAL DE ANALISES CLINICAS", 0, **_NL, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "Rua Exemplo, 123 - Cerqueira Cesar - SP", 0, **_NL, align="C")
        self.ln(10)

    def chapter_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, **_NL, align="L", fill=True)
        self.ln(4)

    def _labeled_value(self, label: str, value: str, ref: str = ""):
        self.set_font("Helvetica", "", 11)
        self.cell(70, 8, label, 0, **_CONT)
        self.set_font("Helvetica", "B", 11)
        self.cell(45, 8, value, 0, **_CONT)
        if ref:
            self.set_font("Helvetica", "I", 10)
            self.cell(0, 8, ref, 0, **_NL)
        else:
            self.ln(8)


def generate_pdf(
    age: int,
    systolic_bp: int,
    diastolic_bp: int,
    bs_mmol: float,
    body_temp_f: float,
    heart_rate: int,
    output_file: str,
    patient_name: Optional[str] = None,
    note: Optional[str] = None,
):
    """Gera um laudo PDF. Recebe valores no formato do modelo (mmol/L, °F)
    e exibe no PDF as unidades clínicas brasileiras (mg/dL, °C)."""
    exam_date = date.today()
    dob = _dob_for_age(age)
    name = patient_name or _random_female_name()
    glucose_mgdl = _mmol_to_mgdl(bs_mmol)
    temp_c = _f_to_c(body_temp_f)

    pdf = _PDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, f"PACIENTE: {name}", 0, **_NL)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"DATA NASC: {_fmt_date_br(dob)} ({age} Anos)      SEXO: FEMININO", 0, **_NL)
    pdf.cell(0, 8, f"DATA DO EXAME: {_fmt_date_br(exam_date)}            PEDIDO: {_random_order_id()}", 0, **_NL)
    if note:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Observacao: {note}", 0, **_NL)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.chapter_title("1. SINAIS VITAIS (Aferição em Repouso)")
    pdf._labeled_value("Pressao Arterial (PA):", f"{systolic_bp}x{diastolic_bp} mmHg", "(Ref: < 120x80)")
    pdf._labeled_value("Frequencia Cardiaca:", f"{heart_rate} bpm", "(Ref: 60 - 100)")
    pdf._labeled_value("Temperatura Axilar:", f"{temp_c} graus C", "(Ref: 36 - 37)")
    pdf.ln(5)

    pdf.chapter_title("2. BIOQUIMICA SANGUINEA")
    pdf._labeled_value("Glicemia de Jejum:", f"{glucose_mgdl} mg/dL", "(Ref: 70 - 99 mg/dL)")
    pdf.ln(5)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "Documento gerado para fins de demonstracao e teste. Nao substitui avaliacao medica profissional.")

    pdf.output(output_file)
    print(f"PDF gerado: {output_file}")
    print(f"   Paciente : {name}, {age} anos")
    print(f"   PA       : {systolic_bp}x{diastolic_bp} mmHg")
    print(f"   FC       : {heart_rate} bpm")
    print(f"   Temp     : {temp_c} graus C  ({body_temp_f} graus F)")
    print(f"   Glicemia : {glucose_mgdl} mg/dL  ({bs_mmol} mmol/L)")


def generate_random_pdf(output_file: str = "laudo_medico_aleatorio.pdf"):
    """Gera um laudo com valores aleatórios sem classificação de risco definida."""
    temp_c = round(random.uniform(36.0, 37.8), 1)
    glucose_mgdl = random.randint(70, 140)
    generate_pdf(
        age=random.randint(18, 45),
        systolic_bp=random.randint(95, 160),
        diastolic_bp=random.randint(60, 105),
        bs_mmol=round(glucose_mgdl / 18.018, 2),
        body_temp_f=round((temp_c * 9 / 5) + 32, 1),
        heart_rate=random.randint(55, 110),
        output_file=output_file,
        note="Valores aleatorios",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gerador de laudos médicos PDF para testes do sistema maternal."
    )
    parser.add_argument("--low", action="store_true", help="Gerar apenas caso de baixo risco")
    parser.add_argument("--high", action="store_true", help="Gerar apenas caso de alto risco")
    parser.add_argument("--random", action="store_true", help="Gerar apenas caso aleatório")
    args = parser.parse_args()

    all_cases = not any([args.low, args.high, args.random])

    if args.low or all_cases:
        c = LOW_RISK_CASE
        print(f"\n[BAIXO RISCO] {c['description']}")
        generate_pdf(
            age=c["Age"], systolic_bp=c["SystolicBP"], diastolic_bp=c["DiastolicBP"],
            bs_mmol=c["BS_mmol"], body_temp_f=c["BodyTemp_f"], heart_rate=c["HeartRate"],
            output_file="laudo_baixo_risco.pdf", note="Caso de referencia - Baixo Risco",
        )

    if args.high or all_cases:
        c = HIGH_RISK_CASE
        print(f"\n[ALTO RISCO] {c['description']}")
        generate_pdf(
            age=c["Age"], systolic_bp=c["SystolicBP"], diastolic_bp=c["DiastolicBP"],
            bs_mmol=c["BS_mmol"], body_temp_f=c["BodyTemp_f"], heart_rate=c["HeartRate"],
            output_file="laudo_alto_risco.pdf", note="Caso de referencia - Alto Risco",
        )

    if args.random or all_cases:
        print("\n[ALEATORIO] Gerando laudo...")
        generate_random_pdf("laudo_medico_exemplo.pdf")


if __name__ == "__main__":
    main()
