from fpdf import FPDF
import random
from datetime import date, datetime, timedelta


def _random_female_name() -> str:
    first_names = [
        "MARIA",
        "ANA",
        "JULIANA",
        "BEATRIZ",
        "FERNANDA",
        "CAMILA",
        "PATRICIA",
        "LUCIANA",
        "CAROLINA",
        "GABRIELA",
        "ISABELA",
        "RAFAELA",
        "VANESSA",
        "AMANDA",
        "TATIANA",
        "LARISSA",
        "BRUNA",
        "RENATA",
        "SABRINA",
        "ALINE",
    ]
    last_names = [
        "SILVA",
        "SANTOS",
        "OLIVEIRA",
        "SOUZA",
        "PEREIRA",
        "COSTA",
        "RODRIGUES",
        "ALMEIDA",
        "NASCIMENTO",
        "LIMA",
        "ARAUJO",
        "FERREIRA",
        "CARVALHO",
        "GOMES",
        "MARTINS",
        "ROCHA",
        "RIBEIRO",
        "ALVES",
    ]

    first = random.choice(first_names)
    middle = random.choice(first_names) if random.random() < 0.35 else None
    last1 = random.choice(last_names)
    last2 = random.choice(last_names)

    parts = [first]
    if middle and middle != first:
        parts.append(middle)
    parts.extend([last1, last2])
    return " ".join(parts)


def _random_date_of_birth(min_age_years: int = 18, max_age_years: int = 60) -> date:
    today = date.today()
    max_birth = today.replace(year=today.year - min_age_years)
    min_birth = today.replace(year=today.year - max_age_years)
    delta_days = (max_birth - min_birth).days
    return min_birth + timedelta(days=random.randint(0, max(0, delta_days)))


def _age_on(dob: date, on_date: date) -> int:
    years = on_date.year - dob.year
    if (on_date.month, on_date.day) < (dob.month, dob.day):
        years -= 1
    return years


def _fmt_date_br(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _random_order_id() -> str:
    return f"#{random.randint(100000, 999999)}"


def generate_random_female_patient() -> dict:
    exam_date = date.today() - timedelta(days=random.randint(0, 30))
    dob = _random_date_of_birth(18, 60)
    return {
        "name": _random_female_name(),
        "sex": "FEMININO",
        "dob": dob,
        "age": _age_on(dob, exam_date),
        "exam_date": exam_date,
        "order_id": _random_order_id(),
    }


def generate_random_vitals() -> dict:
    # Valores plausíveis (não “diagnósticos”), apenas para demo.
    sys = random.randint(95, 160)
    dia = random.randint(60, 105)
    heart_rate = random.randint(55, 110)
    temp_c = round(random.uniform(36.0, 37.8), 1)
    fasting_glucose = random.randint(70, 140)
    return {
        "bp": f"{sys}x{dia} mmHg",
        "hr": f"{heart_rate} bpm",
        "temp": f"{temp_c} °C",
        "glucose": f"{fasting_glucose} mg/dL",
    }

# Criando a classe do PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'LABORATÓRIO CENTRAL DE ANÁLISES CLÍNICAS', 0, 1, 'C')
        self.set_font('Arial', '', 9)
        self.cell(0, 5, 'Rua Exemplo, 123 - Cerqueira César - SP', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

# Instanciando o PDF
pdf = PDF()
pdf.add_page()

patient = generate_random_female_patient()
vitals = generate_random_vitals()

# Dados do Paciente
pdf.set_font('Arial', 'B', 10)
pdf.cell(0, 8, f"PACIENTE: {patient['name']}", 0, 1)
pdf.set_font('Arial', '', 10)
pdf.cell(
    0,
    8,
    f"DATA NASC: {_fmt_date_br(patient['dob'])} ({patient['age']} Anos)      SEXO: {patient['sex']}",
    0,
    1,
)
pdf.cell(0, 8, f"DATA DO EXAME: {_fmt_date_br(patient['exam_date'])}            PEDIDO: {patient['order_id']}", 0, 1)
pdf.ln(5)

# Seção Sinais Vitais
pdf.chapter_title('1. SINAIS VITAIS (Aferição em Repouso)')
pdf.set_font('Arial', '', 11)
# Criando uma "tabela" manual
pdf.cell(60, 8, "Pressão Arterial (PA):", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, vitals["bp"], 0, 0)
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: < 120x80)", 0, 1)

pdf.set_font('Arial', '', 11)
pdf.cell(60, 8, "Frequência Cardíaca:", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, vitals["hr"], 0, 0)
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: 60 - 100)", 0, 1)

pdf.set_font('Arial', '', 11)
pdf.cell(60, 8, "Temperatura Axilar:", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, vitals["temp"], 0, 0) # Em Celsius
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: 36 - 37)", 0, 1)

pdf.ln(5)

# Seção Bioquímica
pdf.chapter_title('2. BIOQUÍMICA SANGUÍNEA')
pdf.set_font('Arial', '', 11)
pdf.cell(60, 8, "Glicemia de Jejum:", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, vitals["glucose"], 0, 0)
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: 70 - 99 mg/dL)", 0, 1)

pdf.output('laudo_medico_exemplo.pdf')
print("PDF gerado com sucesso!")