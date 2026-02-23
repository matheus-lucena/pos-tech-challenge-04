import os
import threading
import time
import wave
from datetime import datetime

import gradio as gr

from config.constants import (
    AUDIO_PLAYER_POLL_INTERVAL,
    SAMPLE_RATE,
    TEMP_AUDIO_DIR,
    TRANSCRIPT_POLL_INTERVAL,
    WAV_CHANNELS,
    WAV_SAMPLEWIDTH,
)
from ui.processors import process_analysis, process_maternal_beats, process_pdf_fill
from ui.realtime_processor import RealtimeAudioProcessor, _realtime_processor

def create_interface():
    with gr.Blocks(
        title="Maternal Health Analysis System",
        theme=gr.themes.Soft()
    ) as demo:
        _add_header()
        
        with gr.Row():
            with gr.Column(scale=1):
                _add_biometric_inputs()
            
            with gr.Column(scale=1):
                _add_audio_inputs()
        
        btn_process, output = _add_action_button()
                
        _add_footer()
    
    return demo


def _add_header():
    gr.Markdown(
        """
        # 🏥 Multimodal Maternal Health Analysis System
        
        This system uses AI to analyze biometric data and consultation audio, 
        providing a complete assessment of maternal health risk.
        
        **Built with CrewAI, AWS SageMaker and AWS Transcribe**
        """
    )


def _add_biometric_inputs():
    gr.Markdown("### 📊 Dados Biométricos")

    age = gr.Number(
        label="Idade",
        value=35,
        minimum=15,
        maximum=50,
        step=1,
        info="Idade da paciente em anos"
    )
    
    systolic_bp = gr.Number(
        label="Pressão Sistólica (mmHg)",
        value=140,
        minimum=80,
        maximum=200,
        step=1,
        info="Pressão arterial sistólica"
    )
    
    diastolic_bp = gr.Number(
        label="Pressão Diastólica (mmHg)",
        value=90,
        minimum=50,
        maximum=150,
        step=1,
        info="Pressão arterial diastólica"
    )
    
    glucose = gr.Number(
        label="Glicemia (BS)",
        value=13.0,
        minimum=3.0,
        maximum=20.0,
        step=0.1,
        info="Nível de açúcar no sangue"
    )
    
    temperature = gr.Number(
        label="Temperatura Corporal (°F)",
        value=98.0,
        minimum=95.0,
        maximum=105.0,
        step=0.1,
        info="Temperatura corporal em Fahrenheit"
    )
    
    heart_rate = gr.Number(
        label="Frequência Cardíaca (bpm)",
        value=70,
        minimum=40,
        maximum=150,
        step=1,
        info="Batimentos por minuto"
    )
    
    return (
        age, systolic_bp, diastolic_bp,
        glucose, temperature, heart_rate
    )


def _add_audio_inputs():
    gr.Markdown("### 🎤 Análise de Áudio (Opcional)")

    audio_file = gr.File(
        label="Upload de Arquivo de Áudio",
        file_types=["audio"],
        type="filepath"
    )
    
    gr.Markdown("**OU**")
    
    gr.Markdown(
        """
        **Opções:**
        - 📤 **Upload de arquivo**: O arquivo será enviado automaticamente para S3
        """
    )
    
    return audio_file


def _add_action_button():
    btn_process = gr.Button(
        "🚀 Iniciar Análise",
        variant="primary",
        size="lg"
    )
    
    output = gr.Markdown(
        label="Resultado da Análise",
        value="Aguardando análise...",
        elem_classes=["resultado-analise"]
    )
    
    return btn_process, output


def _add_footer():
    gr.Markdown(
        """
        ---
        ### ℹ️ Informações
        
        - **Análise Biométrica**: Utiliza modelo XGBoost no AWS SageMaker
        - **Análise de Áudio**: Utiliza AWS Transcribe para transcrição e análise emocional
        - **Sintetização**: Agente médico consolida todas as análises em um relatório final
        """
    )


def create_interface_v2():
    with gr.Blocks(
        title="Sistema de Análise de Saúde Materna",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # 🏥 Sistema de Análise Multimodal de Saúde Materna
            
            Este sistema utiliza IA para analisar dados biométricos e áudios de consultas, 
            fornecendo uma avaliação completa do risco de saúde materna.
            
            **Desenvolvido com CrewAI, AWS SageMaker e AWS Transcribe**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🤰 Análise de Sinal Materno (PCG)")
                gr.Markdown("*Baseado no banco de dados SUFHSDB*")
                maternal_audio_file = gr.File(
                    label="Upload de Arquivo de Áudio Materno (PCG)",
                    file_types=["audio"],
                    type="filepath",
                )
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Upload de Exame Médico (PDF) - Pré-preenchimento")
                pdf_file = gr.File(
                    label="Upload de PDF de Exame Médico",
                    file_types=[".pdf"],
                    type="filepath",
                )
                btn_process_pdf = gr.Button(
                    "📋 Processar PDF e Pré-preencher",
                    variant="secondary",
                    size="lg"
                )
                pdf_status = gr.Markdown(
                    value="",
                    visible=True
                )

        maternal_result = gr.Markdown(
            label="🤰 Análise rápida de sinal materno (PCG)",
            value="Aguardando análise rápida de sinal materno (PCG)..."
        )

        with gr.Row():
            with gr.Column(scale=1):
                age, systolic_bp, diastolic_bp, glucose, temperature, heart_rate = _add_biometric_inputs()

        btn_process_pdf.click(
            fn=process_pdf_fill,
            inputs=[pdf_file],
            outputs=[
                age, systolic_bp, diastolic_bp,
                glucose, temperature, heart_rate, pdf_status
            ],
            show_progress="full"
        )

        btn_process = gr.Button(
            "🚀 Iniciar Análise",
            variant="primary",
            size="lg"
        )
        
        output = gr.Markdown(
            label="Resultado da Análise",
            value="Aguardando análise...",
            elem_classes=["resultado-analise"]
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎤 Tempo Real")
                recording_audio_path = gr.State(value=None)

                def get_audio_devices():
                    devices = RealtimeAudioProcessor.list_audio_devices()
                    if not devices:
                        return ["No devices found"]
                    return [f"{idx}: {name}" for idx, name in devices]

                initial_devices = get_audio_devices()
                device_dropdown = gr.Dropdown(
                    label="Microfone",
                    choices=initial_devices,
                    value=initial_devices[0] if initial_devices else None,
                )
                status_realtime = gr.Markdown(value="", visible=True)
                audio_player = gr.Audio(
                    label="Áudio gravado",
                    type="filepath",
                    visible=True,
                    interactive=False,
                    sources=[],
                )
                transcript_realtime = gr.Textbox(
                    label="Transcrição em Tempo Real",
                    placeholder="A transcrição aparecerá aqui enquanto você fala...",
                    lines=8,
                    interactive=False,
                )
                violence_alert_realtime = gr.Markdown(value="", visible=True, elem_classes=["violence-alert"])
                btn_start_realtime = gr.Button("🎙️ Iniciar transcrição", variant="primary", size="lg")
                btn_stop_realtime = gr.Button("⏹️ Parar", variant="stop", visible=False)
                streaming_state = gr.State(value=False)

                def get_device_index(device_str):
                    if not device_str or ":" not in device_str:
                        return None
                    try:
                        return int(device_str.split(":")[0])
                    except Exception:
                        return None

                def start_realtime(device_selected):
                    if _realtime_processor.is_processing:
                        return (
                            "⚠️ Já existe uma transcrição em andamento.",
                            None,
                            "Aguardando transcrição...",
                            gr.update(visible=True),
                            gr.update(visible=False),
                            None,
                            "",
                            True,
                            None,
                        )
                    device_index = get_device_index(device_selected)
                    def process_stream():
                        try:
                            _realtime_processor.start_microphone_streaming(device_index=device_index)
                        except Exception as e:
                            print(f"Stream error: {e}")
                    thread = threading.Thread(target=process_stream, daemon=True)
                    thread.start()
                    status_msg = (
                        '<div style="padding: 15px; background: #d4edda; border-radius: 8px; '
                        'margin-bottom: 15px; border-left: 4px solid #28a745;">'
                        '<p style="margin: 0; color: #155724;"><strong>🎙️ Gravando...</strong> '
                        'Comece a falar! A transcrição aparecerá em tempo real.</p></div>'
                    )
                    return (
                        status_msg,
                        None,
                        "Aguardando transcrição...",
                        gr.update(visible=True),
                        gr.update(visible=False),
                        None,
                        "",
                        True,
                        None,
                    )

                def stop_realtime():
                    status = _realtime_processor.stop_transcription()
                    transcript = _realtime_processor.get_current_transcript()
                    audio_path = _realtime_processor.get_recorded_audio_path()
                    status_msg = (
                        '<div style="padding: 15px; background: #fff3cd; border-radius: 8px; '
                        'margin-bottom: 15px; border-left: 4px solid #ffc107;">'
                        f'<p style="margin: 0; color: #856404;"><strong>⏹️ {status}</strong></p></div>'
                    )
                    alert_at_stop = _realtime_processor.get_violence_alert()
                    alert_md = (
                        f'<div style="padding: 12px; background: #f8d7da; border-radius: 8px; '
                        f'border-left: 4px solid #dc3545; margin-top: 8px;">'
                        f'<strong>🚨 Alerta de violência:</strong> {alert_at_stop}</div>'
                    ) if alert_at_stop else ""
                    return (
                        status_msg,
                        audio_path if audio_path else None,
                        transcript if transcript else "Nenhuma transcrição capturada.",
                        gr.update(visible=False),
                        gr.update(visible=True),
                        audio_path if audio_path else None,
                        alert_md,
                        False,
                        audio_path,
                    )

                def _violence_alert_md():
                    alert = _realtime_processor.get_violence_alert()
                    if not alert:
                        return ""
                    return (
                        f'<div style="padding: 12px; background: #f8d7da; border-radius: 8px; '
                        f'border-left: 4px solid #dc3545; margin-top: 8px;">'
                        f'<strong>🚨 Alerta de violência detectado:</strong> {alert}</div>'
                    )

                def update_transcript_loop():
                    transcript = _realtime_processor.get_current_transcript()
                    alert_md = _violence_alert_md()
                    yield transcript if transcript else "Aguardando transcrição...", alert_md
                    while _realtime_processor.is_processing:
                        time.sleep(TRANSCRIPT_POLL_INTERVAL)
                        transcript = _realtime_processor.get_current_transcript()
                        alert_md = _violence_alert_md()
                        yield transcript if transcript else "Aguardando transcrição...", alert_md
                    final_transcript = _realtime_processor.get_current_transcript()
                    yield final_transcript if final_transcript else "Transcrição finalizada.", _violence_alert_md()

                def update_audio_player_loop():
                    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
                    while _realtime_processor.is_processing:
                        if _realtime_processor.recorded_audio_frames:
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                temp_file = os.path.join(TEMP_AUDIO_DIR, f"realtime_{timestamp}.wav")
                                wf = wave.open(temp_file, "wb")
                                wf.setnchannels(WAV_CHANNELS)
                                wf.setsampwidth(WAV_SAMPLEWIDTH)
                                wf.setframerate(SAMPLE_RATE)
                                for frame in _realtime_processor.recorded_audio_frames:
                                    wf.writeframes(frame)
                                wf.close()
                                if hasattr(update_audio_player_loop, "last_temp_file"):
                                    try:
                                        if os.path.exists(update_audio_player_loop.last_temp_file):
                                            os.remove(update_audio_player_loop.last_temp_file)
                                    except Exception:
                                        pass
                                update_audio_player_loop.last_temp_file = temp_file
                                yield temp_file
                            except Exception as e:
                                print(f"Error creating temporary audio: {e}")
                                yield gr.update()
                        else:
                            yield gr.update()
                        time.sleep(AUDIO_PLAYER_POLL_INTERVAL)
                    final_audio_path = _realtime_processor.get_recorded_audio_path()
                    if final_audio_path:
                        if hasattr(update_audio_player_loop, "last_temp_file"):
                            try:
                                if os.path.exists(update_audio_player_loop.last_temp_file):
                                    os.remove(update_audio_player_loop.last_temp_file)
                            except Exception:
                                pass
                        yield final_audio_path
                    else:
                        yield gr.update()

                start_event = btn_start_realtime.click(
                    fn=start_realtime,
                    inputs=[device_dropdown],
                    outputs=[
                        status_realtime, audio_player, transcript_realtime,
                        btn_stop_realtime, btn_start_realtime, audio_player,
                        violence_alert_realtime, streaming_state, recording_audio_path,
                    ],
                )
                start_event.then(
                    fn=update_transcript_loop,
                    outputs=[transcript_realtime, violence_alert_realtime],
                )
                start_event.then(
                    fn=update_audio_player_loop,
                    outputs=[audio_player],
                )
                btn_stop_realtime.click(
                    fn=stop_realtime,
                    outputs=[
                        status_realtime, audio_player, transcript_realtime,
                        btn_stop_realtime, btn_start_realtime, audio_player,
                        violence_alert_realtime, streaming_state, recording_audio_path,
                    ],
                )

        maternal_audio_file.change(
            fn=process_maternal_beats,
            inputs=[maternal_audio_file],
            outputs=[maternal_result, heart_rate],
            show_progress="full",
        )
        btn_process.click(
            fn=process_analysis,
            inputs=[
                age, systolic_bp, diastolic_bp,
                glucose, temperature, heart_rate,
                recording_audio_path,
                maternal_audio_file,
            ],
            outputs=output,
            show_progress="full",
        )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ Informações
            
            - **Pré-preenchimento de PDF**: Utiliza AWS Textract para extrair dados de exames médicos em PDF
            - **Análise Biométrica**: Utiliza modelo XGBoost no AWS SageMaker
            - **Análise de Áudio**: Utiliza AWS Transcribe para transcrição e análise emocional
            - **Análise Materna**: Processa sinais de PCG (fonocardiograma) para extrair MHR e classificar risco materno
            - **Sintetização**: Agente médico consolida todas as análises (biométrica, emocional e materna) em um relatório final
            """
        )
    
    return demo

