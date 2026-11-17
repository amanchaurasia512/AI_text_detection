import os
from dotenv import load_dotenv


def load_config() -> None:
    load_dotenv()


def minimum_text_chars() -> int:
    try:
        return max(1, int(os.getenv("AUDIT_MIN_PARAGRAPH_CHARS", "80")))
    except ValueError:
        return 80
