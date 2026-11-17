import json
from pathlib import Path
from ai_text_audit.detectors.mock import MockDetector
from ai_text_audit.pipeline import analyze_document
from ai_text_audit.reporters.json_report import write_json
from ai_text_audit.reporters.markdown_report import write_markdown
from ai_text_audit.segmentation import make_paragraphs

ROOT = Path(__file__).parent


def test_all_text_fixtures_pipeline_and_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_MIN_PARAGRAPH_CHARS", "1")
    for name in ("human.txt", "ai_generated.txt", "mixed.txt"):
        # Text fixtures are exercised as paragraph inputs; CLI documents are PDF/TEX only.
        paragraphs = make_paragraphs((ROOT / "test_cases" / name).read_text(), 1, "fixture")
        assert len(paragraphs) >= 2
        assert all(MockDetector().analyze(p.extracted_text).status == "success" for p in paragraphs)
    assert json.loads((ROOT / "test_cases" / "manifest.json").read_text())["human.txt"]["ground_truth"] is False


def test_short_text_and_report_writers(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_MIN_PARAGRAPH_CHARS", "80")
    source = tmp_path / "short.tex"
    source.write_text("\\documentclass{article}\n\\begin{document}\nToo short.\n\\end{document}", encoding="utf8")
    report = analyze_document(source, MockDetector())
    paragraph = report.document.pages[0].paragraphs[0]
    assert paragraph.detection_result.status == "insufficient_text"
    write_json(report, tmp_path / "report.json")
    write_markdown(report, tmp_path / "report.md")
    summary = json.loads((tmp_path / "report.json").read_text())["document_summary"]
    assert summary["skipped_paragraphs"] >= 1
    assert "Extracted content and detector results" in (tmp_path / "report.md").read_text()


def test_uncertain_classification_is_not_flagged_as_ai():
    from ai_text_audit.models import Document, Page, Paragraph
    from ai_text_audit.models import DetectionResult
    from ai_text_audit.pipeline import build_report

    page = Page(page_number=1, paragraphs=[
        Paragraph(paragraph_id="a", page_number=1, paragraph_number=1,
                  extracted_text="unclear", extraction_method="test",
                  detection_result=DetectionResult(status="success", classification="uncertain")),
        Paragraph(paragraph_id="b", page_number=1, paragraph_number=2,
                  extracted_text="generated", extraction_method="test",
                  detection_result=DetectionResult(status="success", classification="AI-generated")),
    ])
    report = build_report(Document(filename="sample.pdf", document_type="pdf", pages=[page]), "test")
    assert report.page_summaries[0]["flagged_paragraphs"] == 1


def test_heading_and_graphic_labels_are_retained_but_not_counted_as_paragraphs():
    from ai_text_audit.models import Document, Page, Paragraph
    from ai_text_audit.pipeline import build_report

    page = Page(page_number=1, paragraphs=[
        Paragraph(paragraph_id="h", page_number=1, paragraph_number=1,
                  extracted_text="1 Introduction", extraction_method="test", content_type="heading"),
        Paragraph(paragraph_id="g", page_number=1, paragraph_number=2,
                  extracted_text="D/headings", extraction_method="test", content_type="graphic_label"),
        Paragraph(paragraph_id="p", page_number=1, paragraph_number=3,
                  extracted_text="A body paragraph remains part of the paragraph count.", extraction_method="test"),
    ])
    report = build_report(Document(filename="sample.pdf", document_type="pdf", pages=[page]), "mock")
    assert report.document_summary["number_of_paragraphs"] == 1
    assert report.document_summary["number_of_detection_units"] == 1
    assert report.document_summary["number_of_headings"] == 1
    assert report.document_summary["number_of_graphic_labels"] == 1
