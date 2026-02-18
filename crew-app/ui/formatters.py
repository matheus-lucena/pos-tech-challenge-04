import json
import re
from typing import Any, Dict


def format_result(result: Any) -> str:
    try:
        if isinstance(result, str):
            result = _parse_string_result(result)
        
        if hasattr(result, 'raw'):
            result = result.raw
        elif hasattr(result, 'tasks_output'):
            if result.tasks_output:
                result = result.tasks_output[-1]
        elif hasattr(result, '__dict__'):
            result = result.__dict__
        
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
    
    if clean_result.startswith('{') and clean_result.endswith('}'):
        try:
            return json.loads(clean_result)
        except json.JSONDecodeError:
            try:
                start = clean_result.find("{")
                end = clean_result.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = clean_result[start:end]
                    return json.loads(json_str)
            except:
                try:
                    json_match = re.search(r'\{.*\}', clean_result, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                except:
                    pass
    
    return result_str


def _format_dict_result(result: Dict[str, Any]) -> str:
    output = '<div style="font-family: Arial, sans-serif; line-height: 1.6;">\n'
    output += _format_header()
    
    if 'biometric_analysis' in result or 'analise_biometrica' in result:
        key = 'biometric_analysis' if 'biometric_analysis' in result else 'analise_biometrica'
        output += _format_section(
            key,
            '🔬',
            'Biometric Analysis',
            result[key],
            '#007bff'
        )
    
    if 'emotional_analysis' in result or 'analise_emocional' in result:
        key = 'emotional_analysis' if 'emotional_analysis' in result else 'analise_emocional'
        output += _format_section(
            key,
            '🎤',
            'Emotional Analysis',
            result[key],
            '#28a745'
        )
    
    if 'maternal_analysis' in result or 'analise_materna' in result:
        key = 'maternal_analysis' if 'maternal_analysis' in result else 'analise_materna'
        output += _format_section(
            key,
            '🤰',
            'Maternal Analysis',
            result[key],
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
        '<h1 style="margin: 0; font-size: 24px;">📋 Maternal Health Report</h1>\n'
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
        f'<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; '
        f'margin-bottom: 15px; border-left: 4px solid {color};">\n'
        f'<h3 style="margin-top: 0; color: {color}; display: flex; align-items: center; gap: 8px;">\n'
        f'<span>{emoji}</span> <span>{title}</span>\n'
        f'</h3>\n'
        f'<p style="margin-bottom: 0; color: #333;">{content}</p>\n'
        f'</div>\n\n'
    )


def _format_risk_section(risk: str) -> str:
    is_high_risk = "HIGH" in risk.upper() or "ALTO" in risk.upper()
    border_color = "#dc3545" if is_high_risk else "#28a745"
    text_color = "#dc3545" if is_high_risk else "#28a745"
    emoji = "🔴" if is_high_risk else "🟢"
    bg_color = "#fff5f5" if is_high_risk else "#f0fff4"
    
    return (
        f'<div style="background: {bg_color}; padding: 15px; border-radius: 8px; '
        f'margin-bottom: 15px; border-left: 4px solid {border_color};">\n'
        f'<h3 style="margin-top: 0; color: {text_color}; display: flex; align-items: center; gap: 8px;">\n'
        f'<span>{emoji}</span> <span>Final Risk</span>\n'
        f'</h3>\n'
        f'<p style="margin-bottom: 0; color: #333; font-weight: 600; font-size: 16px;">{risk}</p>\n'
        f'</div>\n\n'
    )


def _format_recommendations(recommendations: Any) -> str:
    output = (
        '<div style="background: #fff9e6; padding: 15px; border-radius: 8px; '
        'margin-bottom: 15px; border-left: 4px solid #ffc107;">\n'
        '<h3 style="margin-top: 0; color: #856404; display: flex; align-items: center; gap: 8px;">\n'
        '<span>💡</span> <span>Recommendations</span>\n'
        '</h3>\n'
        '<ul style="margin-bottom: 0; padding-left: 20px; color: #333;">\n'
    )
    
    if isinstance(recommendations, list):
        for rec in recommendations:
            output += f'<li style="margin-bottom: 8px;">{rec}</li>\n'
    else:
        output += f'<li>{recommendations}</li>\n'
    
    output += '</ul>\n</div>\n\n'
    return output
