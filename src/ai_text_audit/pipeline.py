from pathlib import Path
import logging
import re
from datetime import datetime, timezone
from statistics import mean

from ai_text_audit.config import minimum_text_chars
from ai_text_audit.classification import classify_document
from ai_text_audit.detectors.base import BaseDetector
from ai_text_audit.extractors.pdf import PDFExtractor
from ai_text_audit.extractors.tex import TexExtractor
from ai_text_audit.models import Document, DocumentReport
from ai_text_audit.reconstruction import reconstruct_document

log = logging.getLogger(__name__)


def analyze_document(path: Path, detector: BaseDetector, *, ocr: bool = False) -> DocumentReport:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        document = PDFExtractor().extract(path, ocr=ocr)
    elif suffix == ".tex":
        document = TexExtractor().extract(path, ocr=ocr)
    else:
        raise ValueError("Input must be a .pdf or .tex document.")
    classify_document(document)
    reconstruct_document(document)
    apply_analysis_eligibility(document, detector, minimum_text_chars())
    return build_report(document, detector.name)


def apply_analysis_eligibility(document: Document, detector: BaseDetector, minimum: int) -> None:
    """Assign processing status by category and length, then call the detector only for prose."""
    for page in document.pages:
        for paragraph in page.paragraphs:
            if paragraph.reconstruction_status == "review":
                paragraph.processing_status = "reconstruction_review"
                paragraph.detection_result.status = "manual_review"
                paragraph.detection_result.error = paragraph.reconstruction_reason
                continue
            if paragraph.category == "prose" and len(paragraph.extracted_text) < minimum:
                paragraph.processing_status = "manual_review"
                paragraph.detection_result.status = "insufficient_text"
                paragraph.detection_result.error = f"Fewer than {minimum} characters."
                continue
            if paragraph.category == "prose":
                paragraph.processing_status = "eligible"
            elif paragraph.category == "footnote":
                paragraph.processing_status = "manual_review"
                paragraph.detection_result.status = "not_analyzed_content_type"
                paragraph.detection_result.error = "Footnote retained; review eligibility before analysis."
                continue
            elif paragraph.category == "unknown":
                paragraph.processing_status = "manual_review"
                paragraph.detection_result.status = "not_analyzed_content_type"
                paragraph.detection_result.error = "Content category is uncertain; manual review required."
                continue
            else:
                paragraph.processing_status = "retained_non_prose"
                paragraph.detection_result.status = "not_analyzed_content_type"
                paragraph.detection_result.error = f"{paragraph.category} retained; not analyzed as prose."
                continue
            try:
                paragraph.detection_result = detector.analyze(paragraph.extracted_text)
            except Exception as exc:
                log.exception("Detector failed for paragraph %s", paragraph.paragraph_id)
                from ai_text_audit.models import DetectionResult
                paragraph.detection_result = DetectionResult(status="failed", error=f"Detector failed ({type(exc).__name__}).")


def build_report(document: Document, provider: str) -> DocumentReport:
    paragraphs = [p for page in document.pages for p in page.paragraphs]
    prose_paragraphs = [p for p in paragraphs if p.category == "prose"]
    detector_units = [p for p in prose_paragraphs
                      if p.processing_status in {"eligible", "success", "pending"}]
    results = [p.detection_result for p in paragraphs]
    success = sum(p.detection_result.status == "success" for p in prose_paragraphs)
    failed = sum(p.detection_result.status == "failed" for p in prose_paragraphs)
    skipped = sum(p.detection_result.status == "insufficient_text" for p in prose_paragraphs)
    block_success = sum(r.status == "success" for r in results)
    block_failed = sum(r.status == "failed" for r in results)
    block_skipped = sum(r.status == "insufficient_text" for r in results)
    limitations = ["Detector outputs are indicators and do not prove AI authorship.",
                   "Scores are provider-defined and must not be treated as calibrated probabilities unless documented by that provider."]
    if provider == "mock":
        limitations.append("Mock scores are synthetic and do not assess authorship.")
    if not document.exact_page_mapping:
        limitations.append("Exact PDF page mapping is unavailable for this source document.")
    if document.document_type in {"pdf", "tex"} and document.pages:
        limitations.append(
            "Text embedded only as vector shapes or raster image content may not be recoverable. "
            "OCR fallback runs only for pages with no usable text layer when enabled."
        )
        limitations.append(
            "Tables and figures are retained as extracted text blocks; visual cell, spatial, and chart relationships are not reconstructed."
        )
        limitations.append(
            "PDF font encodings can make some accented glyphs or ligatures imperfect; extracted text is not a proofread transcription."
        )
    page_summaries = []
    for page in document.pages:
        scored = [p.detection_result.score for p in page.paragraphs
                  if p.detection_result.status == "success" and p.detection_result.score is not None]
        page_summaries.append({"page_number": page.page_number,
                               "paragraph_count": sum(p.category == "prose" for p in page.paragraphs),
                               "analyzed_paragraphs": sum(
                                   p.category == "prose" and p.detection_result.status == "success"
                                   for p in page.paragraphs
                               ),
                               "skipped_paragraphs": sum(
                                   p.category == "prose" and p.detection_result.status == "insufficient_text"
                                   for p in page.paragraphs
                               ),
                               "heading_count": sum(p.category == "heading" for p in page.paragraphs),
                               "caption_count": sum(p.category in {"figure_caption", "table_caption"} for p in page.paragraphs),
                               "list_item_count": sum(p.content_type == "list_item" for p in page.paragraphs),
                               "graphic_label_count": sum(p.category in {"chart_label", "diagram_label"} for p in page.paragraphs),
                               "average_provider_score": mean(scored) if scored else None,
                               "flagged_paragraphs": sum(
                                   _is_ai_flagged(p.detection_result.classification)
                                   for p in page.paragraphs if p.category == "prose"
                               ),
                               "unclassified_paragraphs": sum(
                                   p.category == "prose"
                                   and p.detection_result.status == "success"
                                   and not p.detection_result.classification
                                   for p in page.paragraphs
                               ),
                               "not_analyzed_content_blocks": sum(
                                   p.category != "prose" for p in page.paragraphs
                               )})
    # Cross-paragraph averaging is deliberately omitted absent provider semantics.
    category_counts = {category: sum(p.category == category for p in paragraphs)
                       for category in ("prose", "heading", "figure_caption", "table_caption",
                                        "table_content", "diagram_label", "chart_label", "reference",
                                        "footnote", "equation", "header_footer", "unknown")}
    status_counts = {status: sum(p.processing_status == status for p in paragraphs)
                     for status in ("eligible", "insufficient_text", "retained_non_prose",
                                    "manual_review", "excluded_by_category", "reconstruction_review")}
    reconstructed = [p for p in paragraphs
                     if p.reconstruction_status in {"reconstructed_join", "reconstructed_split"}]
    reconstructed_source_ids = {source_id for p in reconstructed for source_id in p.source_block_ids}
    summary = {"filename": document.filename, "document_type": document.document_type,
               "extraction_status": document.extraction_status,
               "compilation_status": document.compilation_status,
               "number_of_pages": sum(page.page_number is not None for page in document.pages),
               "number_of_paragraphs": len(prose_paragraphs),
               "total_reconstructed_paragraphs": len(paragraphs),
               "number_of_reconstructed_paragraphs": len(reconstructed),
               "number_of_source_blocks": len(document.source_blocks),
               "source_blocks_in_reconstruction": len(reconstructed_source_ids),
               "number_of_manual_review_items": status_counts["manual_review"] + status_counts["reconstruction_review"],
               "number_of_short_prose_items": sum(p.category == "prose" and len(p.extracted_text) < minimum_text_chars() for p in paragraphs),
               "number_of_retained_non_prose_items": sum(p.processing_status == "retained_non_prose" for p in paragraphs),
               "content_category_counts": category_counts,
               "processing_status_counts": status_counts,
               "reconstruction_decision_count": len(document.reconstruction_decisions),
               "number_of_content_blocks": len(paragraphs),
               "number_of_detection_units": len(detector_units),
               "number_of_headings": sum(p.category == "heading" for p in paragraphs),
               "number_of_captions": sum(p.category in {"figure_caption", "table_caption"} for p in paragraphs),
               "number_of_list_items": sum(p.content_type == "list_item" for p in paragraphs),
               "number_of_graphic_labels": sum(
                   p.category in {"diagram_label", "chart_label"} or p.content_type == "graphic_label"
                   for p in paragraphs
               ),
               "successfully_analyzed_paragraphs": success,
               "failed_paragraphs": failed,
               "skipped_paragraphs": skipped,
               "successfully_analyzed_content_blocks": block_success,
               "failed_content_blocks": block_failed,
               "skipped_content_blocks": block_skipped,
               "not_analyzed_content_blocks": sum(p.category != "prose" for p in paragraphs),
               "extraction_failure_count": len(document.extraction_failures),
               "document_level_score": None,
               "document_level_summary": "No document-level score calculated; aggregation semantics are provider-dependent."}
    return DocumentReport(document=document, provider=provider,
                          generated_at=datetime.now(timezone.utc).isoformat(), limitations=limitations,
                          document_summary=summary, page_summaries=page_summaries)


def _is_ai_flagged(classification: str | None) -> bool:
    if not classification:
        return False
    normalized = classification.strip().lower().replace("_", " ").replace("-", " ")
    if any(term in normalized for term in ("uncertain", "unknown", "ambiguous", "indeterminate")):
        return False
    return bool(re.search(r"\bai\b|machine generated|likely ai|probably ai", normalized))
