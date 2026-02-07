"""Exemplo de uso do Comprehend Medical para análise de texto médico."""

from services.comprehend_medical_service import ComprehendMedicalService

# Exemplo de texto médico transcrito
texto_exemplo = """
Paciente: Maria Silva, 28 anos, gestante de 32 semanas.
Queixa principal: Dor de cabeça e pressão alta.
Exame físico: Pressão arterial 150/95 mmHg, frequência cardíaca 88 bpm.
Diagnóstico: Hipertensão gestacional.
Medicação prescrita: Metildopa 250mg, 2x ao dia.
Próxima consulta: 15/12/2024.
"""

def main():
    """Exemplo de análise com Comprehend Medical."""
    print("=== EXEMPLO DE ANÁLISE COM COMPREHEND MEDICAL ===\n")
    print(f"Texto a ser analisado:\n{texto_exemplo}\n")
    print("=" * 60 + "\n")
    
    try:
        # Inicializa o serviço
        service = ComprehendMedicalService()
        
        # Realiza análise completa
        print("Analisando texto...\n")
        analysis = service.analyze_text(texto_exemplo)
        
        # Formata e exibe resultados
        formatted_result = service.format_analysis_result(analysis)
        print(formatted_result)
        
        # Exibe também o resultado em JSON para referência
        print("\n" + "=" * 60)
        print("RESULTADO EM JSON (para referência):")
        print("=" * 60)
        import json
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        print("\nCertifique-se de que:")
        print("1. As credenciais AWS estão configuradas")
        print("2. A região AWS está correta (padrão: us-east-1)")
        print("3. O serviço Comprehend Medical está habilitado na sua conta AWS")

if __name__ == "__main__":
    main()

