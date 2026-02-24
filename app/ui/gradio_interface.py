import gradio as gr

from ui.processors import process_analysis, process_maternal_beats, process_pdf_fill
from ui.realtime_processor import RealtimeAudioProcessor
from ui.realtime_handlers import (
    start_realtime,
    stop_realtime,
    update_transcript_loop,
    update_audio_player_loop,
)


def create_interface_v2():
    with gr.Blocks(
        title="Sistema de Análise de Saúde Materna",
        theme=gr.themes.Soft(),
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
                    size="lg",
                )
                pdf_status = gr.Markdown(value="", visible=True)

        maternal_result = gr.Markdown(
            label="🤰 Análise rápida de sinal materno (PCG)",
            value="Aguardando análise rápida de sinal materno (PCG)...",
        )

        with gr.Row():
            with gr.Column(scale=1):
                age, systolic_bp, diastolic_bp, glucose, temperature, heart_rate = (
                    _add_biometric_inputs()
                )

        btn_process_pdf.click(
            fn=process_pdf_fill,
            inputs=[pdf_file],
            outputs=[age, systolic_bp, diastolic_bp, glucose, temperature, heart_rate, pdf_status],
            show_progress="full",
        )

        btn_process = gr.Button("🚀 Iniciar Análise", variant="primary", size="lg")
        output = gr.Markdown(
            label="Resultado da Análise",
            value="Aguardando análise...",
            elem_classes=["resultado-analise"],
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎤 Tempo Real")
                recording_audio_path = gr.State(value=None)

                initial_devices = [
                    f"{idx}: {name}"
                    for idx, name in RealtimeAudioProcessor.list_audio_devices()
                ] or ["No devices found"]

                device_dropdown = gr.Dropdown(
                    label="Microfone",
                    choices=initial_devices,
                    value=initial_devices[0],
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
                violence_alert_realtime = gr.Markdown(
                    value="", visible=True, elem_classes=["violence-alert"]
                )
                btn_start_realtime = gr.Button(
                    "🎙️ Iniciar transcrição", variant="primary", size="lg"
                )
                btn_stop_realtime = gr.Button("⏹️ Parar", variant="stop", visible=False)
                streaming_state = gr.State(value=False)

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


def _add_biometric_inputs():
    gr.Markdown("### 📊 Dados Biométricos")

    age = gr.Number(
        label="Idade", value=35, minimum=15, maximum=50, step=1,
        info="Idade da paciente em anos",
    )
    systolic_bp = gr.Number(
        label="Pressão Sistólica (mmHg)", value=140, minimum=80, maximum=200, step=1,
        info="Pressão arterial sistólica",
    )
    diastolic_bp = gr.Number(
        label="Pressão Diastólica (mmHg)", value=90, minimum=50, maximum=150, step=1,
        info="Pressão arterial diastólica",
    )
    glucose = gr.Number(
        label="Glicemia (BS)", value=13.0, minimum=3.0, maximum=20.0, step=0.1,
        info="Nível de açúcar no sangue",
    )
    temperature = gr.Number(
        label="Temperatura Corporal (°F)", value=98.0, minimum=95.0, maximum=105.0, step=0.1,
        info="Temperatura corporal em Fahrenheit",
    )
    heart_rate = gr.Number(
        label="Frequência Cardíaca (bpm)", value=70, minimum=40, maximum=150, step=1,
        info="Batimentos por minuto",
    )

    return age, systolic_bp, diastolic_bp, glucose, temperature, heart_rate
