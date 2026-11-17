
import argparse
import logging
import os
from pathlib import Path

from ai_text_audit.config import load_config
from ai_text_audit.detectors import APIDetector, MockDetector, OllamaDetector
from ai_text_audit.pipeline import analyze_document
from ai_text_audit.reporters import write_json, write_markdown


def main() -> int:
    parser = argparse.ArgumentParser(prog="ai-text-audit")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a PDF or LaTeX document")
    analyze.add_argument("document", type=Path)
    analyze.add_argument(
        "--provider",
        choices=["mock", "api", "ollama"],
        default="mock",
    )
    analyze.add_argument("--output", type=Path, default=Path("outputs"))
    analyze.add_argument("--ocr", action="store_true")
    analyze.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("pdfminer").setLevel(logging.WARNING)

    load_config()

    try:
        if args.provider == "mock":
            detector = MockDetector()

        elif args.provider == "api":
            detector = APIDetector(
                timeout=float(os.getenv("AUDIT_API_TIMEOUT", "30")),
                retries=int(os.getenv("AUDIT_API_RETRIES", "2")),
                text_field=os.getenv("AUDIT_API_TEXT_FIELD", "text"),
                auth_prefix=os.getenv(
                    "AUDIT_API_AUTH_PREFIX", "Bearer "
                ).rstrip() + " ",
            )

        else:
            detector = OllamaDetector(
                model=os.getenv("AUDIT_OLLAMA_MODEL", "qwen2.5:3b"),
                host=os.getenv(
                    "AUDIT_OLLAMA_HOST",
                    "http://localhost:11434",
                ),
                timeout=float(os.getenv("AUDIT_OLLAMA_TIMEOUT", "120")),
            )

        report = analyze_document(args.document, detector, ocr=args.ocr)

        args.output.mkdir(parents=True, exist_ok=True)
        write_json(report, args.output / "report.json")
        write_markdown(report, args.output / "report.md")

        logging.info("Reports written to %s", args.output)
        return 0

    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
