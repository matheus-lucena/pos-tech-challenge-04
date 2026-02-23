import json
import re
from typing import Any, Dict


def normalize_result(result: Any) -> Any:
    if hasattr(result, "raw"):
        return result.raw
    if hasattr(result, "tasks_output") and result.tasks_output:
        return result.tasks_output[-1]
    if hasattr(result, "__dict__"):
        return result.__dict__
    return result


def parse_result_str(result_str: str) -> Any:
    return _parse_string_result(result_str)


def format_result(result: Any) -> str:
    try:
        result = normalize_result(result)
        if isinstance(result, str):
            result = _parse_string_result(result)
        if isinstance(result, dict):
            return _format_dict_result(result)
        return f'<div style="padding: 20px; background: #f8f9fa; border-radius: 8px;"><pre>{str(result)}</pre></div>'
    except Exception as e:
        return (
            f'<div style="padding: 20px; background: #fff5f5; border-radius: 8px; color: #dc3545;">'
            f'<strong>Error formatting result:</strong><br>{str(e)}<br><br>'
            f'<pre>{str(result)}</pre></div>'
        )


def _parse_string_result(result_str: str) -> Any:
    clean_result = result_str.strip()
    if clean_result.startswith('```json'):
        clean_result = re.sub(r'^```json\s*', '', clean_result, flags=re.MULTILINE)
        clean_result = re.sub(r'\s*```\s*$', '', clean_result, flags=re.MULTILINE)
        clean_result = clean_result.strip()
    elif clean_result.startswith('```'):
        clean_result = re.sub(r'^```\w*\s*', '', clean_result, flags=re.MULTILINE)
        clean_result = re.sub(r'\s*```\s*$', '', clean_result, flags=re.MULTILINE)
        clean_result = clean_result.strip()
    
    if clean_result.startswith('{'):
        try:
            return json.loads(clean_result)
        except json.JSONDecodeError:
            try:
                start = clean_result.find("{")
                end = clean_result.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = clean_result[start:end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    json_match = re.search(r'\{.*\}', clean_result, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
    
    return result_str


def _format_dict_result(result: Dict[str, Any]) -> str:
    output = '<div style="font-family: Arial, sans-serif; line-height: 1.6;">\n'
    output += _format_header()
    if 'maternal_analysis' in result or 'analise_materna' in result:
        key = 'maternal_analysis' if 'maternal_analysis' in result else 'analise_materna'
        content = result[key]
        output += _format_section(
            key,
            '🤰',
            'Análise Materna',
            content,
            '#17a2b8'
        )
    
    if 'final_risk' in result or 'risco_final' in result:
        key = 'final_risk' if 'final_risk' in result else 'risco_final'
        output += _format_risk_section(result[key])
    
    if 'recommendations' in result or 'recomendacoes' in result:
        key = 'recommendations' if 'recommendations' in result else 'recomendacoes'
        output += _format_recommendations(result[key])
    
    output += '</div>'
    return output


def _format_header() -> str:
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
    return (
        f'<div style="background: #ffffff; padding: 20px; border-radius: 8px; '
        f'margin-bottom: 20px; border-left: 4px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">\n'
        f'<h3 style="margin-top: 0; margin-bottom: 15px; color: {color}; font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px;">\n'
        f'<span>{emoji}</span> <span>{title}</span>\n'
        f'</h3>\n'
        f'<p style="margin-bottom: 0; line-height: 1.8; color: #333;">{content}</p>\n'
        f'</div>\n\n'
    )


def _format_risk_section(risk: str) -> str:
    is_high_risk = "HIGH" in risk.upper() or "ALTO" in risk.upper()
    border_color = "#dc3545" if is_high_risk else "#28a745"
    text_color = "#dc3545" if is_high_risk else "#28a745"
    emoji = "🔴" if is_high_risk else "🟢"
    bg_color = "#fff5f5" if is_high_risk else "#f0fff4"
    
    return (
        f'<div style="background: {bg_color}; padding: 20px; border-radius: 8px; '
        f'margin-bottom: 20px; border-left: 4px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">\n'
        f'<h3 style="margin-top: 0; margin-bottom: 15px; color: {text_color}; font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px;">\n'
        f'<span>{emoji}</span> <span>Risco Final</span>\n'
        f'</h3>\n'
        f'<p style="margin-bottom: 0; color: #333; font-weight: 700; font-size: 20px; line-height: 1.6;">{risk}</p>\n'
        f'</div>\n\n'
    )


def _format_recommendations(recommendations: Any) -> str:
    output = (
        '<div style="background: #fff9e6; padding: 20px; border-radius: 8px; '
        'margin-bottom: 20px; border-left: 4px solid #ffc107; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">\n'
        '<h3 style="margin-top: 0; margin-bottom: 15px; color: #856404; font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px;">\n'
        '<span>💡</span> <span>Recomendações</span>\n'
        '</h3>\n'
        '<ul style="margin-bottom: 0; padding-left: 25px; color: #333; line-height: 1.8;">\n'
    )
    
    if isinstance(recommendations, list):
        for rec in recommendations:
            output += f'<li style="margin-bottom: 12px; padding-left: 5px;">{rec}</li>\n'
    else:
        output += f'<li style="padding-left: 5px;">{recommendations}</li>\n'
    
    output += '</ul>\n</div>\n\n'
    return output
