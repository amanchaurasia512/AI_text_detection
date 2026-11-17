import argparse
import logging
import os
from pathlib import Path

from ai_text_audit.config import load_config
from ai_text_audit.detectors import APIDetector, MockDetector
from ai_text_audit.pipeline import analyze_document
from ai_text_audit.reporters import write_json, write_markdown


def main() -> int:
    parser = argparse.ArgumentParser(prog="ai-text-audit")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Analyze a PDF or LaTeX document")
    analyze.add_argument("document", type=Path)
    analyze.add_argument("--provider", default="mock")
    analyze.add_argument("--output", type=Path, default=Path("outputs"))
    analyze.add_argument("--ocr", action="store_true")
    analyze.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # pdfminer emits per-token traces at DEBUG; keep verbose mode actionable.
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    load_config()
    try:
        detector = MockDetector() if args.provider == "mock" else APIDetector(
            timeout=float(os.getenv("AUDIT_API_TIMEOUT", "30")),
            retries=int(os.getenv("AUDIT_API_RETRIES", "2")),
            text_field=os.getenv("AUDIT_API_TEXT_FIELD", "text"),
            auth_prefix=os.getenv("AUDIT_API_AUTH_PREFIX", "Bearer ").rstrip() + " ",
        ) if args.provider == "api" else None
        if detector is None:
            parser.error("Unknown provider. Use 'mock' or 'api'.")
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
