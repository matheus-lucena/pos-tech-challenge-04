"""
Aplicação principal do Sistema de Análise Multimodal de Saúde Materna.

Este módulo orquestra a interface Gradio e inicializa o sistema.
"""

from ui import criar_interface


def main():
    """
    Função principal que inicializa e executa a interface Gradio.
    """
    demo = criar_interface()
    demo.launch(
        server_name="0.0.0.0",  # Permite acesso externo
        server_port=7860,        # Porta padrão do Gradio
        share=False              # Mude para True se quiser criar link público temporário
    )


if __name__ == "__main__":
    main()
