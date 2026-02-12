"""Interface Gradio para o sistema de análise."""

import gradio as gr
from ui.processors import processar_analise, processar_pdf_preenchimento


def criar_interface():
    """
    Cria e retorna a interface Gradio.
    
    Returns:
        Interface Gradio configurada
    """
    with gr.Blocks(
        title="Sistema de Análise de Saúde Materna",
        theme=gr.themes.Soft()
    ) as demo:
        _add_header()
        
        with gr.Row():
            with gr.Column(scale=1):
                _add_biometric_inputs()
            
            with gr.Column(scale=1):
                _add_audio_inputs()
        
        btn_processar, output = _add_action_button()
                
        _add_footer()
    
    return demo


def _add_header():
    """Adiciona o cabeçalho da interface."""
    gr.Markdown(
        """
        # 🏥 Sistema de Análise Multimodal de Saúde Materna
        
        Este sistema utiliza IA para analisar dados biométricos e áudios de consultas, 
        fornecendo uma avaliação completa do risco de saúde materna.
        
        **Desenvolvido com CrewAI, AWS SageMaker e AWS Transcribe**
        """
    )


def _add_biometric_inputs():
    """Adiciona os inputs de dados biométricos."""
    gr.Markdown("### 📊 Dados Biométricos")
    
    idade = gr.Number(
        label="Idade",
        value=35,
        minimum=15,
        maximum=50,
        step=1,
        info="Idade da paciente em anos"
    )
    
    pressao_sistolica = gr.Number(
        label="Pressão Sistólica (mmHg)",
        value=140,
        minimum=80,
        maximum=200,
        step=1,
        info="Pressão arterial sistólica"
    )
    
    pressao_diastolica = gr.Number(
        label="Pressão Diastólica (mmHg)",
        value=90,
        minimum=50,
        maximum=150,
        step=1,
        info="Pressão arterial diastólica"
    )
    
    glicemia = gr.Number(
        label="Glicemia (BS)",
        value=13.0,
        minimum=3.0,
        maximum=20.0,
        step=0.1,
        info="Nível de açúcar no sangue"
    )
    
    temperatura = gr.Number(
        label="Temperatura Corporal (°F)",
        value=98.0,
        minimum=95.0,
        maximum=105.0,
        step=0.1,
        info="Temperatura corporal em Fahrenheit"
    )
    
    frequencia_cardiaca = gr.Number(
        label="Frequência Cardíaca (bpm)",
        value=70,
        minimum=40,
        maximum=150,
        step=1,
        info="Batimentos por minuto"
    )
    
    return (
        idade, pressao_sistolica, pressao_diastolica,
        glicemia, temperatura, frequencia_cardiaca
    )


def _add_audio_inputs():
    """Adiciona os inputs de áudio."""
    gr.Markdown("### 🎤 Análise de Áudio (Opcional)")
    
    arquivo_audio = gr.File(
        label="Upload de Arquivo de Áudio",
        file_types=["audio"],
        type="filepath"
    )
    
    gr.Markdown("**OU**")
    
    s3_audio = gr.Textbox(
        label="Caminho S3 do Áudio (Alternativa)",
        placeholder="s3://bucket-name/audio-file.mp3",
        info="Se o arquivo já estiver no S3, informe o caminho completo",
        lines=2
    )
    
    gr.Markdown(
        """
        **Opções:**
        - 📤 **Upload de arquivo**: O arquivo será enviado automaticamente para S3
        - 🔗 **Caminho S3**: Use se o arquivo já estiver no bucket
        
        **Exemplo de caminho S3:** `s3://fiap-pos-fase04-matheuslucena/vitima-01.mp3`
        """
    )
    
    return arquivo_audio, s3_audio


def _add_action_button():
    """Adiciona o botão de ação e área de output."""
    btn_processar = gr.Button(
        "🚀 Iniciar Análise",
        variant="primary",
        size="lg"
    )
    
    output = gr.Markdown(
        label="Resultado da Análise",
        value="Aguardando análise...",
        elem_classes=["resultado-analise"]
    )
    
    return btn_processar, output


def _add_footer():
    """Adiciona o rodapé com informações."""
    gr.Markdown(
        """
        ---
        ### ℹ️ Informações
        
        - **Análise Biométrica**: Utiliza modelo XGBoost no AWS SageMaker
        - **Análise de Áudio**: Utiliza AWS Transcribe para transcrição e análise emocional
        - **Sintetização**: Agente médico consolida todas as análises em um relatório final
        """
    )


# Versão alternativa que funciona melhor com Gradio
def criar_interface_v2():
    """
    Cria e retorna a interface Gradio (versão alternativa mais funcional).
    
    Returns:
        Interface Gradio configurada
    """
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
        
        # Seção de upload de PDF para pré-preenchimento
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Upload de Exame Médico (PDF) - Pré-preenchimento")
                gr.Markdown(
                    """
                    **Faça upload de um PDF de exame médico para preencher automaticamente os campos abaixo.**
                    
                    O sistema extrairá automaticamente:
                    - Idade
                    - Pressão Arterial (Sistólica/Diastólica)
                    - Glicemia
                    - Temperatura
                    - Frequência Cardíaca
                    """
                )
                
                arquivo_pdf = gr.File(
                    label="Upload de PDF de Exame Médico",
                    file_types=[".pdf"],
                    type="filepath",
                )
                
                btn_processar_pdf = gr.Button(
                    "📋 Processar PDF e Pré-preencher",
                    variant="secondary",
                    size="lg"
                )
                
                status_pdf = gr.Markdown(
                    value="",
                    visible=True
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Dados Biométricos")
                
                idade = gr.Number(
                    label="Idade",
                    value=35,
                    minimum=15,
                    maximum=50,
                    step=1,
                    info="Idade da paciente em anos"
                )
                
                pressao_sistolica = gr.Number(
                    label="Pressão Sistólica (mmHg)",
                    value=140,
                    minimum=80,
                    maximum=200,
                    step=1,
                    info="Pressão arterial sistólica"
                )
                
                pressao_diastolica = gr.Number(
                    label="Pressão Diastólica (mmHg)",
                    value=90,
                    minimum=50,
                    maximum=150,
                    step=1,
                    info="Pressão arterial diastólica"
                )
                
                glicemia = gr.Number(
                    label="Glicemia (BS)",
                    value=13.0,
                    minimum=3.0,
                    maximum=20.0,
                    step=0.1,
                    info="Nível de açúcar no sangue"
                )
                
                temperatura = gr.Number(
                    label="Temperatura Corporal (°F)",
                    value=98.0,
                    minimum=95.0,
                    maximum=105.0,
                    step=0.1,
                    info="Temperatura corporal em Fahrenheit"
                )
                
                frequencia_cardiaca = gr.Number(
                    label="Frequência Cardíaca (bpm)",
                    value=70,
                    minimum=40,
                    maximum=150,
                    step=1,
                    info="Batimentos por minuto"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Análise de Áudio de Consulta (Opcional)")
                
                arquivo_audio = gr.File(
                    label="Upload de Arquivo de Áudio (Consulta/Emocional)",
                    file_types=["audio"],
                    type="filepath"
                )
                
                gr.Markdown("**OU**")
                
                s3_audio = gr.Textbox(
                    label="Caminho S3 do Áudio (Alternativa)",
                    placeholder="s3://bucket-name/audio-file.mp3",
                    info="Se o arquivo já estiver no S3, informe o caminho completo",
                    lines=2
                )
                
                gr.Markdown("---")
                gr.Markdown("### 👶 Análise de Sinal Fetal (PCG) - Opcional")
                gr.Markdown("*Baseado no banco de dados SUFHSDB*")
                
                arquivo_audio_fetal = gr.File(
                    label="Upload de Arquivo de Áudio Fetal (PCG)",
                    file_types=["audio"],
                    type="filepath",
                )
                
                gr.Markdown("**OU**")
                
                s3_audio_fetal = gr.Textbox(
                    label="Caminho S3 do Áudio Fetal (Alternativa)",
                    placeholder="s3://bucket-name/fetal-pcg.wav",
                    info="Caminho S3 do arquivo de PCG fetal",
                    lines=2
                )
                
                gr.Markdown(
                    """
                    **Opções:**
                    - 📤 **Upload de arquivo**: O arquivo será enviado automaticamente para S3
                    - 🔗 **Caminho S3**: Use se o arquivo já estiver no bucket
                    
                    **Análise Fetal:**
                    - Extrai Frequência Cardíaca Fetal (FHR)
                    - Detecta bradicardia, taquicardia e variabilidade
                    - Classifica risco fetal em tempo real
                    """
                )
        
        # Conecta o evento de processamento de PDF
        btn_processar_pdf.click(
            fn=processar_pdf_preenchimento,
            inputs=[arquivo_pdf],
            outputs=[
                idade, pressao_sistolica, pressao_diastolica,
                glicemia, temperatura, frequencia_cardiaca, status_pdf
            ],
            show_progress="full"
        )
        
        btn_processar = gr.Button(
            "🚀 Iniciar Análise",
            variant="primary",
            size="lg"
        )
        
        output = gr.Markdown(
            label="Resultado da Análise",
            value="Aguardando análise...",
            elem_classes=["resultado-analise"]
        )
        
        btn_processar.click(
            fn=processar_analise,
            inputs=[
                idade, pressao_sistolica, pressao_diastolica,
                glicemia, temperatura, frequencia_cardiaca, 
                arquivo_audio, s3_audio,
                arquivo_audio_fetal, s3_audio_fetal
            ],
            outputs=output,
            show_progress="full"
        )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ Informações
            
            - **Pré-preenchimento de PDF**: Utiliza AWS Textract para extrair dados de exames médicos em PDF
            - **Análise Biométrica**: Utiliza modelo XGBoost no AWS SageMaker
            - **Análise de Áudio**: Utiliza AWS Transcribe para transcrição e análise emocional
            - **Análise Fetal**: Processa sinais de PCG (fonocardiograma) para extrair FHR e classificar risco fetal
            - **Sintetização**: Agente médico consolida todas as análises (biométrica, emocional e fetal) em um relatório final
            """
        )
    
    return demo

