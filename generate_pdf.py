from fpdf import FPDF

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

# Dados do Paciente
pdf.set_font('Arial', 'B', 10)
pdf.cell(0, 8, "PACIENTE: JOÃO DA SILVA", 0, 1)
pdf.set_font('Arial', '', 10)
pdf.cell(0, 8, "DATA NASC: 12/02/1989 (35 Anos)      SEXO: MASCULINO", 0, 1)
pdf.cell(0, 8, "DATA DO EXAME: 12/02/2024            PEDIDO: #998877", 0, 1)
pdf.ln(5)

# Seção Sinais Vitais
pdf.chapter_title('1. SINAIS VITAIS (Aferição em Repouso)')
pdf.set_font('Arial', '', 11)
# Criando uma "tabela" manual
pdf.cell(60, 8, "Pressão Arterial (PA):", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, "140x90 mmHg", 0, 0)
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: < 120x80)", 0, 1)

pdf.set_font('Arial', '', 11)
pdf.cell(60, 8, "Frequência Cardíaca:", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, "70 bpm", 0, 0)
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: 60 - 100)", 0, 1)

pdf.set_font('Arial', '', 11)
pdf.cell(60, 8, "Temperatura Axilar:", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, "36.7 °C", 0, 0) # Note que está em Celsius
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: 36 - 37)", 0, 1)

pdf.ln(5)

# Seção Bioquímica
pdf.chapter_title('2. BIOQUÍMICA SANGUÍNEA')
pdf.set_font('Arial', '', 11)
pdf.cell(60, 8, "Glicemia de Jejum:", 0, 0)
pdf.set_font('Arial', 'B', 11)
pdf.cell(40, 8, "95 mg/dL", 0, 0)
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 8, "(Ref: 70 - 99 mg/dL)", 0, 1)

pdf.output('laudo_medico_exemplo.pdf')
print("PDF gerado com sucesso!")