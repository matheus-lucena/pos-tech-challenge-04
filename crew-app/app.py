from ui import create_interface
from dotenv import load_dotenv

load_dotenv()


def main():
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
