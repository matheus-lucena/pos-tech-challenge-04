"""
Demo de Inferência - API de Risco de Saúde Materna
===================================================
Este script demonstra o uso da API de predição de risco de saúde materna
usando um modelo XGBoost implantado no Amazon SageMaker.

Autor: Sistema de Predição de Risco Materno
Data: 2026
"""

import boto3
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "sagemaker-xgboost-2026-01-22-21-24-26-641")
REGION = os.getenv("AWS_REGION", "us-east-1")

try:
    client = boto3.client("sagemaker-runtime", region_name=REGION)
    print("✓ Cliente SageMaker inicializado com sucesso")
except Exception as e:
    print(f"✗ Erro ao inicializar cliente SageMaker: {e}")
    exit(1)

# ============================================================================
# CASOS DE TESTE - Cenários Realistas
# ============================================================================
test_cases: List[Dict[str, Any]] = [
    {
        "name": "Paciente 1 - Baixo Risco",
        "description": "Mulher jovem, pressão normal, glicemia controlada",
        "data": {
            "Age": 25,
            "SystolicBP": 110,
            "DiastolicBP": 70,
            "BS": 6.5,
            "BodyTemp": 98.0,
            "HeartRate": 70
        },
        "expected_risk": "Baixo"
    },
    {
        "name": "Paciente 2 - Baixo Risco",
        "description": "Mulher jovem, parâmetros vitais dentro da normalidade",
        "data": {
            "Age": 28,
            "SystolicBP": 115,
            "DiastolicBP": 75,
            "BS": 7.0,
            "BodyTemp": 98.2,
            "HeartRate": 75
        },
        "expected_risk": "Baixo"
    },
    {
        "name": "Paciente 3 - Baixo Risco",
        "description": "Mulher adulta, sinais vitais estáveis",
        "data": {
            "Age": 30,
            "SystolicBP": 118,
            "DiastolicBP": 78,
            "BS": 7.2,
            "BodyTemp": 98.5,
            "HeartRate": 72
        },
        "expected_risk": "Baixo"
    },
    {
        "name": "Paciente 4 - Alto Risco",
        "description": "Mulher mais velha, hipertensão, glicemia elevada",
        "data": {
            "Age": 40,
            "SystolicBP": 150,
            "DiastolicBP": 100,
            "BS": 10.5,
            "BodyTemp": 99.5,
            "HeartRate": 95
        },
        "expected_risk": "Alto"
    },
    {
        "name": "Paciente 5 - Alto Risco",
        "description": "Mulher com múltiplos fatores de risco",
        "data": {
            "Age": 42,
            "SystolicBP": 160,
            "DiastolicBP": 105,
            "BS": 12.0,
            "BodyTemp": 100.0,
            "HeartRate": 100
        },
        "expected_risk": "Alto"
    },
    {
        "name": "Paciente 6 - Caso Limítrofe",
        "description": "Caso intermediário para testar sensibilidade do modelo",
        "data": {
            "Age": 35,
            "SystolicBP": 130,
            "DiastolicBP": 85,
            "BS": 8.5,
            "BodyTemp": 98.8,
            "HeartRate": 85
        },
        "expected_risk": "Indeterminado"
    }
]

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def print_header(text: str, width: int = 80):
    """Imprime um cabeçalho formatado"""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def print_section(text: str, width: int = 80):
    """Imprime uma seção formatada"""
    print("\n" + "-" * width)
    print(f"  {text}")
    print("-" * width)

def format_risk_result(result: Dict) -> str:
    """Formata o resultado de risco de forma visual"""
    is_high_risk = result.get("maternal_health_risk", False)
    probability = result.get("risk_probability", "0%")
    
    if is_high_risk:
        risk_label = "🔴 ALTO RISCO"
        risk_color = "ALTO"
    else:
        risk_label = "🟢 BAIXO RISCO"
        risk_color = "BAIXO"
    
    return f"{risk_label} | Probabilidade: {probability}"

def validate_input(data: Dict) -> bool:
    """Valida os dados de entrada"""
    required_fields = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    
    for field in required_fields:
        if field not in data:
            print(f"✗ Campo obrigatório ausente: {field}")
            return False
        if not isinstance(data[field], (int, float)):
            print(f"✗ Campo {field} deve ser numérico")
            return False
    
    return True

def print_patient_info(case: Dict, index: int, total: int):
    """Imprime informações do paciente de forma formatada"""
    print_header(f"TESTE {index}/{total}: {case['name']}", 80)
    print(f"\n📋 Descrição: {case['description']}")
    print(f"\n📊 Dados do Paciente:")
    print(f"   • Idade: {case['data']['Age']} anos")
    print(f"   • Pressão Sistólica: {case['data']['SystolicBP']} mmHg")
    print(f"   • Pressão Diastólica: {case['data']['DiastolicBP']} mmHg")
    print(f"   • Glicemia (BS): {case['data']['BS']} mmol/L")
    print(f"   • Temperatura Corporal: {case['data']['BodyTemp']} °F")
    print(f"   • Frequência Cardíaca: {case['data']['HeartRate']} bpm")

# ============================================================================
# FUNÇÃO PRINCIPAL DE TESTE
# ============================================================================

def test_inference(test_case: Dict, index: int, total: int) -> Dict[str, Any]:
    """Executa um teste de inferência e retorna o resultado"""
    print_patient_info(test_case, index, total)
    
    if not validate_input(test_case['data']):
        return {"success": False, "error": "Dados inválidos"}
    
    start_time = time.time()
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(test_case['data'])
        )
        
        elapsed_time = time.time() - start_time
        
        result = json.loads(response['Body'].read().decode())
        
        print(f"\n{format_risk_result(result)}")
        print(f"\n⏱️  Tempo de resposta: {elapsed_time:.3f} segundos")
        
        if 'expected_risk' in test_case:
            expected = test_case['expected_risk']
            actual = "Alto" if result.get("maternal_health_risk") else "Baixo"
            if expected != "Indeterminado":
                match = "✓" if (expected == actual) else "✗"
                print(f"\n{match} Expectativa: {expected} | Resultado: {actual}")
        
        return {
            "success": True,
            "result": result,
            "elapsed_time": elapsed_time,
            "case_name": test_case['name']
        }
        
    except client.exceptions.ModelError as e:
        print(f"\n✗ ERRO DO MODELO: {e}")
        return {"success": False, "error": str(e), "case_name": test_case['name']}
    except client.exceptions.ValidationError as e:
        print(f"\n✗ ERRO DE VALIDAÇÃO: {e}")
        return {"success": False, "error": str(e), "case_name": test_case['name']}
    except Exception as e:
        print(f"\n✗ ERRO INESPERADO: {e}")
        return {"success": False, "error": str(e), "case_name": test_case['name']}

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal da demo"""
    print_header("DEMO: API DE PREDIÇÃO DE RISCO DE SAÚDE MATERNA", 80)
    print(f"\n📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Região AWS: {REGION}")
    print(f"🔗 Endpoint: {ENDPOINT_NAME}")
    print(f"📦 Total de casos de teste: {len(test_cases)}")
    
    results = []
    total_time = 0
    high_risk_count = 0
    low_risk_count = 0
    
    print_header("INICIANDO TESTES DE INFERÊNCIA", 80)
    
    for i, test_case in enumerate(test_cases, 1):
        result = test_inference(test_case, i, len(test_cases))
        results.append(result)
        
        if result.get("success"):
            total_time += result.get("elapsed_time", 0)
            if result["result"].get("maternal_health_risk"):
                high_risk_count += 1
            else:
                low_risk_count += 1
        
        if i < len(test_cases):
            time.sleep(0.5)
    
    # Resumo final
    print_header("RESUMO DA DEMONSTRAÇÃO", 80)
    
    successful_tests = sum(1 for r in results if r.get("success"))
    failed_tests = len(results) - successful_tests
    
    print(f"\n📊 Estatísticas Gerais:")
    print(f"   • Testes executados: {len(results)}")
    print(f"   • Testes bem-sucedidos: {successful_tests} ✓")
    print(f"   • Testes com erro: {failed_tests} {'✗' if failed_tests > 0 else ''}")
    
    if successful_tests > 0:
        print(f"\n🎯 Resultados de Risco:")
        print(f"   • Casos de Alto Risco: {high_risk_count}")
        print(f"   • Casos de Baixo Risco: {low_risk_count}")
        print(f"\n⏱️  Performance:")
        print(f"   • Tempo total: {total_time:.3f} segundos")
        print(f"   • Tempo médio por requisição: {total_time/successful_tests:.3f} segundos")
    
    print_section("DETALHES DOS RESULTADOS", 80)
    for i, result in enumerate(results, 1):
        if result.get("success"):
            case_name = result.get("case_name", f"Teste {i}")
            risk_result = result["result"]
            risk_status = "ALTO RISCO" if risk_result.get("maternal_health_risk") else "BAIXO RISCO"
            probability = risk_result.get("risk_probability", "N/A")
            print(f"\n{i}. {case_name}")
            print(f"   Status: {risk_status} | Probabilidade: {probability}")
        else:
            case_name = result.get("case_name", f"Teste {i}")
            error = result.get("error", "Erro desconhecido")
            print(f"\n{i}. {case_name}")
            print(f"   ✗ Erro: {error}")
    
    print_header("DEMO CONCLUÍDA", 80)
    print("\n✅ Demonstração finalizada com sucesso!\n")

if __name__ == "__main__":
    main()