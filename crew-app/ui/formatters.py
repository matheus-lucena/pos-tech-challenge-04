"""Formatadores para exibição de resultados."""

import json
import re
from typing import Any, Dict


def formatar_resultado(resultado: Any) -> str:
    """
    Formata o resultado da análise para exibição na interface.
    
    Args:
        resultado: Resultado da análise (dict, string, ou objeto CrewAI)
    
    Returns:
        String HTML formatada
    """
    try:
        # Se for string, tenta parsear como JSON
        if isinstance(resultado, str):
            resultado = _parse_string_result(resultado)
        
        # Se for um objeto com atributos (como resultado do CrewAI)
        if hasattr(resultado, 'raw'):
            resultado = resultado.raw
        elif hasattr(resultado, 'tasks_output'):
            if resultado.tasks_output:
                resultado = resultado.tasks_output[-1]
        elif hasattr(resultado, '__dict__'):
            resultado = resultado.__dict__
        
        # Se for dict, formata de forma bonita
        if isinstance(resultado, dict):
            return _format_dict_result(resultado)
        
        # Se não conseguir formatar, retorna como string
        return f'<div style="padding: 20px; background: #f8f9fa; border-radius: 8px;"><pre>{str(resultado)}</pre></div>'
    except Exception as e:
        return (
            f'<div style="padding: 20px; background: #fff5f5; border-radius: 8px; color: #dc3545;">'
            f'<strong>Erro ao formatar resultado:</strong><br>{str(e)}<br><br>'
            f'<pre>{str(resultado)}</pre></div>'
        )


def _parse_string_result(resultado_str: str) -> Any:
    """
    Tenta parsear uma string como JSON.
    
    Args:
        resultado_str: String a ser parseada
    
    Returns:
        Objeto parseado ou string original
    """
    resultado_limpo = resultado_str.strip()
    
    # Tenta parsear diretamente se for JSON válido
    if resultado_limpo.startswith('{') and resultado_limpo.endswith('}'):
        try:
            return json.loads(resultado_limpo)
        except json.JSONDecodeError:
            # Tenta extrair JSON de uma string que contém JSON
            try:
                inicio = resultado_limpo.find("{")
                fim = resultado_limpo.rfind("}") + 1
                if inicio >= 0 and fim > inicio:
                    json_str = resultado_limpo[inicio:fim]
                    return json.loads(json_str)
            except:
                # Tenta usar regex para encontrar JSON
                try:
                    json_match = re.search(r'\{.*\}', resultado_limpo, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                except:
                    pass
    
    return resultado_str


def _format_dict_result(resultado: Dict[str, Any]) -> str:
    """
    Formata um dicionário como HTML estilizado.
    
    Args:
        resultado: Dicionário com os resultados
    
    Returns:
        String HTML formatada
    """
    output = '<div style="font-family: Arial, sans-serif; line-height: 1.6;">\n'
    output += _format_header()
    
    # Análise Biométrica
    if 'analise_biometrica' in resultado:
        output += _format_section(
            'analise_biometrica',
            '🔬',
            'Análise Biométrica',
            resultado['analise_biometrica'],
            '#007bff'
        )
    
    # Análise Emocional
    if 'analise_emocional' in resultado:
        output += _format_section(
            'analise_emocional',
            '🎤',
            'Análise Emocional',
            resultado['analise_emocional'],
            '#28a745'
        )
    
    # Risco Final
    if 'risco_final' in resultado:
        output += _format_risk_section(resultado['risco_final'])
    
    # Recomendações
    if 'recomendacoes' in resultado:
        output += _format_recommendations(resultado['recomendacoes'])
    
    output += '</div>'
    return output


def _format_header() -> str:
    """Retorna o cabeçalho HTML do relatório."""
    return (
        '<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
        'color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">\n'
        '<h1 style="margin: 0; font-size: 24px;">📋 Relatório de Saúde Materna</h1>\n'
        '</div>\n\n'
    )


def _format_section(
    key: str,
    emoji: str,
    title: str,
    content: str,
    color: str
) -> str:
    """
    Formata uma seção do relatório.
    
    Args:
        key: Chave da seção
        emoji: Emoji para a seção
        title: Título da seção
        content: Conteúdo da seção
        color: Cor da borda
    
    Returns:
        String HTML formatada
    """
    return (
        f'<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; '
        f'margin-bottom: 15px; border-left: 4px solid {color};">\n'
        f'<h3 style="margin-top: 0; color: {color}; display: flex; align-items: center; gap: 8px;">\n'
        f'<span>{emoji}</span> <span>{title}</span>\n'
        f'</h3>\n'
        f'<p style="margin-bottom: 0; color: #333;">{content}</p>\n'
        f'</div>\n\n'
    )


def _format_risk_section(risco: str) -> str:
    """
    Formata a seção de risco final.
    
    Args:
        risco: Classificação de risco
    
    Returns:
        String HTML formatada
    """
    is_alto_risco = "ALTO" in risco.upper()
    cor_borda = "#dc3545" if is_alto_risco else "#28a745"
    cor_texto = "#dc3545" if is_alto_risco else "#28a745"
    emoji = "🔴" if is_alto_risco else "🟢"
    bg_color = "#fff5f5" if is_alto_risco else "#f0fff4"
    
    return (
        f'<div style="background: {bg_color}; padding: 15px; border-radius: 8px; '
        f'margin-bottom: 15px; border-left: 4px solid {cor_borda};">\n'
        f'<h3 style="margin-top: 0; color: {cor_texto}; display: flex; align-items: center; gap: 8px;">\n'
        f'<span>{emoji}</span> <span>Risco Final</span>\n'
        f'</h3>\n'
        f'<p style="margin-bottom: 0; color: #333; font-weight: 600; font-size: 16px;">{risco}</p>\n'
        f'</div>\n\n'
    )


def _format_recommendations(recomendacoes: Any) -> str:
    """
    Formata a seção de recomendações.
    
    Args:
        recomendacoes: Lista ou string de recomendações
    
    Returns:
        String HTML formatada
    """
    output = (
        '<div style="background: #fff9e6; padding: 15px; border-radius: 8px; '
        'margin-bottom: 15px; border-left: 4px solid #ffc107;">\n'
        '<h3 style="margin-top: 0; color: #856404; display: flex; align-items: center; gap: 8px;">\n'
        '<span>💡</span> <span>Recomendações</span>\n'
        '</h3>\n'
        '<ul style="margin-bottom: 0; padding-left: 20px; color: #333;">\n'
    )
    
    if isinstance(recomendacoes, list):
        for rec in recomendacoes:
            output += f'<li style="margin-bottom: 8px;">{rec}</li>\n'
    else:
        output += f'<li>{recomendacoes}</li>\n'
    
    output += '</ul>\n</div>\n\n'
    return output

