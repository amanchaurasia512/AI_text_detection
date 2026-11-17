from pathlib import Path
from ai_text_audit.models import DocumentReport


def write_markdown(report: DocumentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = report.document_summary
    lines = [f"# AI Text Audit: {s['filename']}", "", f"- Type: {s['document_type']}",
             f"- Extraction status: {s.get('extraction_status', 'unknown')}",
             f"- Compilation status: {s.get('compilation_status') or 'not applicable'}",
             f"- Provider: {report.provider}", f"- Pages: {s['number_of_pages']}",
             f"- Paragraphs: {s['number_of_paragraphs']}",
             f"- Content blocks: {s.get('number_of_content_blocks', s['number_of_paragraphs'])}",
             f"- Headings: {s.get('number_of_headings', 0)}",
             f"- Captions: {s.get('number_of_captions', 0)}",
             f"- List items: {s.get('number_of_list_items', 0)}",
             f"- Graphic labels retained: {s.get('number_of_graphic_labels', 0)}",
             f"- Source blocks: {s.get('number_of_source_blocks', 0)}",
             f"- Reconstructed paragraphs: {s.get('total_reconstructed_paragraphs', s['number_of_paragraphs'])}",
             f"- Reconstructed units: {s.get('number_of_reconstructed_paragraphs', 0)}",
             f"- Manual review items: {s.get('number_of_manual_review_items', 0)}",
             f"- Short prose items: {s.get('number_of_short_prose_items', 0)}",
             f"- Retained non-prose items: {s.get('number_of_retained_non_prose_items', 0)}",
             f"- Analyzed paragraphs: {s['successfully_analyzed_paragraphs']}",
             f"- Failed paragraphs: {s['failed_paragraphs']}",
             f"- Skipped paragraphs (below length threshold): {s['skipped_paragraphs']}", "",
             f"- Analyzed content blocks: {s.get('successfully_analyzed_content_blocks', 0)}",
             f"- Failed content blocks: {s.get('failed_content_blocks', 0)}",
             f"- Skipped content blocks: {s.get('skipped_content_blocks', 0)}",
             f"- Not analyzed (structural) blocks: {s.get('not_analyzed_content_blocks', 0)}",
             f"- Extraction failures: {s.get('extraction_failure_count', len(report.document.extraction_failures))}", "",
             "## Extracted content and detector results", "",
             "| Page(s) | Paragraph | Category | Source block IDs | Provider score | Classification | Processing status | Text |",
             "|---|---|---|---|---:|---|---|---|"]
    for page in report.document.pages:
        for p in page.paragraphs:
            r = p.detection_result
            score = f"{r.score:g}" if r.score is not None else "—"
            classification = (r.classification or "—").replace("|", "\\|")
            page_num = str(p.page_number) if p.page_number is not None else "Unavailable"
            page_nums = ", ".join(map(str, p.source_page_numbers)) or page_num
            text = p.extracted_text.replace("|", "\\|").replace("\n", " ")
            ids = ", ".join(p.source_block_ids).replace("|", "\\|")
            lines.append(f"| {page_nums} | {p.paragraph_id} | {p.category} | {ids} | {score} | {classification} | {p.processing_status} | {text} |")
    lines += ["", "## Page summary", "", "| Page | Paragraphs | Average provider score | Flagged | Unclassified |",
              "|---:|---:|---:|---:|---:|"]
    for p in report.page_summaries:
        score = f"{p['average_provider_score']:.4g}" if p["average_provider_score"] is not None else "—"
        page_label = p["page_number"] if p["page_number"] is not None else "Unavailable"
        lines.append(f"| {page_label} | {p['paragraph_count']} | {score} | {p['flagged_paragraphs']} | {p['unclassified_paragraphs']} |")
    lines += ["", "## Document summary", "", s["document_level_summary"], "",
              "## Uncertainty and limitations", ""]
    lines += [f"- {item}" for item in report.limitations]
    lines += ["", "## Content categories", ""]
    lines += [f"- {key}: {value}" for key, value in s.get("content_category_counts", {}).items()]
    lines += ["", "## Processing statuses", ""]
    lines += [f"- {key}: {value}" for key, value in s.get("processing_status_counts", {}).items()]
    lines += ["", "## Reconstruction decisions", ""]
    for decision in report.document.reconstruction_decisions:
        ids = ", ".join(decision.get("source_block_ids", []))
        pages = ", ".join(map(str, decision.get("page_numbers", [])))
        lines.append(f"- {decision.get('action')} ({'automatic' if decision.get('automatic') else 'manual review'}), blocks {ids}, pages {pages}: {decision.get('reason')}")
    for page in report.document.pages:
        for note in page.content_notes:
            lines.append(f"- Page {page.page_number}: {note}")
    for failure in report.document.extraction_failures:
        lines.append(f"- Extraction: {failure}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
