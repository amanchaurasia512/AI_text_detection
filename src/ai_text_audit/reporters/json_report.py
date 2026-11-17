import json
from pathlib import Path
from ai_text_audit.models import DocumentReport


def write_json(report: DocumentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
