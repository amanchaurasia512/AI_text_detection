from pathlib import Path
from ai_text_audit.models import DocumentReport


def write_markdown(report: DocumentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = report.document_summary
    lines = [f"# AI Text Audit: {s['filename']}", "", f"- Type: {s['document_type']}",
             f"- Provider: {report.provider}", f"- Pages: {s['number_of_pages']}",
             f"- Paragraphs: {s['number_of_paragraphs']}",
             f"- Analyzed: {s['successfully_analyzed_paragraphs']}",
             f"- Failed: {s['failed_paragraphs']}", f"- Skipped: {s['skipped_paragraphs']}", "",
             "## Paragraph results", "", "| Page | Paragraph | AI score | Classification | Status |",
             "|---:|---:|---:|---|---|"]
    for page in report.document.pages:
        for p in page.paragraphs:
            r = p.detection_result
            score = f"{r.score:g}" if r.score is not None else "—"
            classification = (r.classification or "—").replace("|", "\\|")
            page_num = str(p.page_number) if p.page_number is not None else "Unavailable"
            lines.append(f"| {page_num} | {p.paragraph_number} | {score} | {classification} | {r.status} |")
    lines += ["", "## Page summary", "", "| Page | Paragraphs | Average provider score | Flagged | Unclassified |",
              "|---:|---:|---:|---:|---:|"]
    for p in report.page_summaries:
        score = f"{p['average_provider_score']:.4g}" if p["average_provider_score"] is not None else "—"
        page_label = p["page_number"] if p["page_number"] is not None else "Unavailable"
        lines.append(f"| {page_label} | {p['paragraph_count']} | {score} | {p['flagged_paragraphs']} | {p['unclassified_paragraphs']} |")
    lines += ["", "## Document summary", "", s["document_level_summary"], "",
              "## Uncertainty and limitations", ""]
    lines += [f"- {item}" for item in report.limitations]
    for failure in report.document.extraction_failures:
        lines.append(f"- Extraction: {failure}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
