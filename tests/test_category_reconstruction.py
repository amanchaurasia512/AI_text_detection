from ai_text_audit.classification import classify_document
from ai_text_audit.detectors.base import BaseDetector
from ai_text_audit.models import Document, Page, Paragraph, SourceBlock, DetectionResult
from ai_text_audit.pipeline import apply_analysis_eligibility, build_report
from ai_text_audit.reconstruction import reconstruct_document


class CountingDetector(BaseDetector):
    name = "counting"

    def __init__(self):
        self.calls = []

    def analyze(self, text):
        self.calls.append(text)
        return DetectionResult(status="success", score=0.5)


def _block(block_id, text, page=1, legacy="paragraph", **evidence):
    return SourceBlock(source_block_id=block_id, page_number=page, text=text,
                       extraction_method="synthetic", legacy_content_type=legacy,
                       bbox=evidence.pop("bbox", {}), layout_evidence=evidence)


def _paragraph(block_id, text, page=1, legacy="paragraph", category="prose"):
    return Paragraph(paragraph_id=block_id, page_number=page, paragraph_number=1,
                     extracted_text=text, extraction_method="synthetic",
                     content_type=legacy, category=category,
                     source_block_ids=[block_id], source_page_numbers=[page])


def test_p03_72_character_prose_is_retained_for_manual_review():
    text = "Chunking for retrieval. Several studies com- pare many chunking methods."
    assert len(text) == 72
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[
        Page(page_number=1, paragraphs=[_paragraph("short", text)])
    ])
    detector = CountingDetector()
    apply_analysis_eligibility(doc, detector, 80)
    item = doc.pages[0].paragraphs[0]
    assert item.processing_status == "manual_review"
    assert item.detection_result.status == "insufficient_text"
    assert detector.calls == []


def test_short_table_cell_is_classified_and_retained_without_detector_call():
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[
        Page(page_number=1, paragraphs=[_paragraph("cell", "Setting Value", legacy="table_row")])
    ], source_blocks=[_block("cell", "Setting Value", legacy="table_row")])
    classify_document(doc)
    detector = CountingDetector()
    apply_analysis_eligibility(doc, detector, 80)
    item = doc.pages[0].paragraphs[0]
    assert item.category == "table_content"
    assert item.processing_status == "retained_non_prose"
    assert detector.calls == []


def test_diagram_chart_and_caption_categories_use_deterministic_evidence():
    blocks = [
        _block("diagram", "25 documents used for design", legacy="graphic_label", graphic_kind="diagram"),
        _block("chart", "(a) Parser settings", legacy="graphic_label", graphic_kind="chart"),
        _block("fcap", "Figure 2: Results by setting.", legacy="caption"),
        _block("tcap", "Table 3: Short names.", legacy="caption"),
    ]
    paragraphs = [_paragraph(b.source_block_id, b.text, legacy=b.legacy_content_type) for b in blocks]
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[Page(page_number=1, paragraphs=paragraphs)], source_blocks=blocks)
    classify_document(doc)
    assert [p.category for p in doc.pages[0].paragraphs] == ["diagram_label", "chart_label", "figure_caption", "table_caption"]
    assert all(p.classification_reason for p in doc.pages[0].paragraphs)


def test_reference_and_footnote_categories_are_retained():
    refs = [
        _block("heading", "References", legacy="heading"),
        _block("ref", "Jane Doe. 2025. A study. Preprint, arXiv:2501.12345."),
        _block("foot", "https://example.org/paper", bbox={"top": 760}, page_height=792),
    ]
    ps = [_paragraph(b.source_block_id, b.text, legacy=b.legacy_content_type) for b in refs]
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[Page(page_number=1, paragraphs=ps)], source_blocks=refs)
    classify_document(doc)
    assert [p.category for p in doc.pages[0].paragraphs] == ["heading", "reference", "footnote"]


def test_sentence_split_around_footnote_reconstructs_with_provenance():
    blocks = [
        _block("a", "The parser performs text extrac-"),
        _block("foot", "https://example.org", legacy="paragraph"),
        _block("b", "tion before retrieval."),
    ]
    ps = [_paragraph("a", blocks[0].text), _paragraph("foot", blocks[1].text, category="footnote"),
          _paragraph("b", blocks[2].text)]
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[Page(page_number=1, paragraphs=ps)], source_blocks=blocks)
    # Categories may be assigned before reconstruction by the classifier.
    classify_document(doc)
    reconstruct_document(doc)
    prose = [p for p in doc.pages[0].paragraphs if p.category == "prose"]
    assert len(prose) == 1
    assert prose[0].extracted_text == "The parser performs text extraction before retrieval."
    assert prose[0].source_block_ids == ["a", "b"]
    assert any(p.paragraph_id == "foot" for p in doc.pages[0].paragraphs)
    assert doc.reconstruction_decisions[0]["action"] == "join"


def test_paragraph_split_across_pages_keeps_both_page_mappings():
    a = _block("p1", "This paragraph continues across the page", page=1)
    b = _block("p2", "break and finishes here.", page=2)
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[
        Page(page_number=1, paragraphs=[_paragraph("p1", a.text, 1)]),
        Page(page_number=2, paragraphs=[_paragraph("p2", b.text, 2)]),
    ], source_blocks=[a, b])
    classify_document(doc)
    reconstruct_document(doc)
    joined = doc.pages[0].paragraphs[0]
    assert joined.source_page_numbers == [1, 2]
    assert joined.extracted_text == "This paragraph continues across the page break and finishes here."
    assert doc.pages[1].paragraphs == []


def test_p03_page_break_before_numeric_result_reconstructs():
    left = _block("left", "The answer to the comparison is NO-GO: the gap", page=1)
    right = _block("right", "− 0.007 [− 0.045, 0.023], and the recommendation is fixed.", page=2)
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[
        Page(page_number=1, paragraphs=[_paragraph("left", left.text, 1)]),
        Page(page_number=2, paragraphs=[_paragraph("right", right.text, 2)]),
    ], source_blocks=[left, right])
    classify_document(doc)
    reconstruct_document(doc)
    item = doc.pages[0].paragraphs[0]
    assert item.reconstruction_status == "reconstructed_join"
    assert item.source_page_numbers == [1, 2]
    assert "gap − 0.007" in item.extracted_text


def test_distinct_complete_paragraphs_are_not_merged():
    a = _block("a", "This paragraph is complete.")
    b = _block("b", "Another paragraph is independently complete.")
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[Page(page_number=1, paragraphs=[
        _paragraph("a", a.text), _paragraph("b", b.text)
    ])], source_blocks=[a, b])
    classify_document(doc)
    reconstruct_document(doc)
    assert [p.extracted_text for p in doc.pages[0].paragraphs] == [a.text, b.text]


def test_layout_evidence_splits_a_merged_source_block():
    text = "First paragraph ends here. Second paragraph starts here."
    offset = text.index("Second")
    block = _block("merged", text, paragraph_break_offsets=[offset])
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[Page(page_number=1, paragraphs=[
        _paragraph("merged", text)
    ])], source_blocks=[block])
    classify_document(doc)
    reconstruct_document(doc)
    assert [p.extracted_text for p in doc.pages[0].paragraphs] == [
        "First paragraph ends here.", "Second paragraph starts here."
    ]
    assert all(p.source_block_ids == ["merged"] for p in doc.pages[0].paragraphs)
    assert doc.reconstruction_decisions[0]["action"] == "split"


def test_unknown_category_is_manual_review():
    block = _block("u", "Decorative text", legacy="mystery")
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[Page(page_number=1, paragraphs=[
        _paragraph("u", block.text, legacy="mystery")
    ])], source_blocks=[block])
    classify_document(doc)
    apply_analysis_eligibility(doc, CountingDetector(), 1)
    item = doc.pages[0].paragraphs[0]
    assert item.category == "unknown"
    assert item.processing_status == "manual_review"


def test_reconstructed_report_counts_sources_categories_and_review():
    block = _block("short", "A concise passage.")
    doc = Document(filename="fixture.pdf", document_type="pdf", pages=[Page(page_number=1, paragraphs=[
        _paragraph("short", block.text)
    ])], source_blocks=[block])
    classify_document(doc)
    apply_analysis_eligibility(doc, CountingDetector(), 80)
    report = build_report(doc, "counting")
    assert report.document_summary["number_of_source_blocks"] == 1
    assert report.document_summary["content_category_counts"]["prose"] == 1
    assert report.document_summary["processing_status_counts"]["manual_review"] == 1
    assert report.document_summary["number_of_manual_review_items"] == 1


def test_uncertain_classification_is_not_flagged_as_ai():
    item = _paragraph("p", "Some text here.")
    item.detection_result = DetectionResult(status="success", classification="uncertain")
    page = Page(page_number=1, paragraphs=[item])
    report = build_report(Document(filename="fixture.pdf", document_type="pdf", pages=[page]), "mock")
    assert report.page_summaries[0]["flagged_paragraphs"] == 0
