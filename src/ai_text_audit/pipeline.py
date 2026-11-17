from pathlib import Path
import logging
from datetime import datetime, timezone
from statistics import mean

from ai_text_audit.config import minimum_text_chars
from ai_text_audit.detectors.base import BaseDetector
from ai_text_audit.extractors.pdf import PDFExtractor
from ai_text_audit.extractors.tex import TexExtractor
from ai_text_audit.models import Document, DocumentReport

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
    minimum = minimum_text_chars()
    for page in document.pages:
        for paragraph in page.paragraphs:
            if len(paragraph.extracted_text) < minimum:
                paragraph.detection_result.status = "insufficient_text"
                paragraph.detection_result.error = f"Fewer than {minimum} characters."
                continue
            try:
                paragraph.detection_result = detector.analyze(paragraph.extracted_text)
            except Exception as exc:
                log.exception("Detector failed for paragraph %s", paragraph.paragraph_id)
                from ai_text_audit.models import DetectionResult
                paragraph.detection_result = DetectionResult(status="failed", error=f"Detector failed ({type(exc).__name__}).")
    return build_report(document, detector.name)


def build_report(document: Document, provider: str) -> DocumentReport:
    paragraphs = [p for page in document.pages for p in page.paragraphs]
    results = [p.detection_result for p in paragraphs]
    success = sum(r.status == "success" for r in results)
    failed = sum(r.status == "failed" for r in results)
    skipped = sum(r.status == "insufficient_text" for r in results)
    limitations = ["Detector outputs are indicators and do not prove AI authorship.",
                   "Scores are provider-defined and must not be treated as calibrated probabilities unless documented by that provider.",
                   "Mock scores are synthetic and do not assess authorship."]
    if not document.exact_page_mapping:
        limitations.append("Exact PDF page mapping is unavailable for this source document.")
    page_summaries = []
    for page in document.pages:
        scored = [p.detection_result.score for p in page.paragraphs
                  if p.detection_result.status == "success" and p.detection_result.score is not None]
        page_summaries.append({"page_number": page.page_number, "paragraph_count": len(page.paragraphs),
                               "average_provider_score": mean(scored) if scored else None,
                               "flagged_paragraphs": sum("ai" in (p.detection_result.classification or "").lower()
                                                         for p in page.paragraphs),
                               "unclassified_paragraphs": sum(not p.detection_result.classification for p in page.paragraphs)})
    # Cross-paragraph averaging is deliberately omitted absent provider semantics.
    summary = {"filename": document.filename, "document_type": document.document_type,
               "number_of_pages": sum(page.page_number is not None for page in document.pages),
               "number_of_paragraphs": len(paragraphs),
               "successfully_analyzed_paragraphs": success, "failed_paragraphs": failed,
               "skipped_paragraphs": skipped, "document_level_score": None,
               "document_level_summary": "No document-level score calculated; aggregation semantics are provider-dependent."}
    return DocumentReport(document=document, provider=provider,
                          generated_at=datetime.now(timezone.utc).isoformat(), limitations=limitations,
                          document_summary=summary, page_summaries=page_summaries)
