"""Page-aware PDF text extraction with optional OCR fallback."""
from pathlib import Path
import logging

import pdfplumber
from ai_text_audit.models import Document, Page
from .base import BaseExtractor

log = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    def extract(self, path: Path, *, ocr: bool = False) -> Document:
        doc = Document(filename=path.name, document_type="pdf")
        try:
            with pdfplumber.open(path) as pdf:
                for index, pdf_page in enumerate(pdf.pages, start=1):
                    text = (pdf_page.extract_text(layout=False) or "").strip()
                    method = "pdfplumber"
                    paragraph_texts = _positioned_paragraphs(pdf_page) if text else []
                    if not text and ocr:
                        try:
                            import fitz
                            import pytesseract
                            from PIL import Image
                            import io
                            with fitz.open(path) as rendered_pdf:
                                image = rendered_pdf[index - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                            text = pytesseract.image_to_string(Image.open(io.BytesIO(image.tobytes("png")))).strip()
                            method = "ocr:tesseract"
                            paragraph_texts = [part.strip() for part in text.split("\n\n") if part.strip()]
                        except Exception as exc:
                            log.warning("OCR failed on page %d (%s)", index, type(exc).__name__)
                            doc.extraction_failures.append(f"Page {index}: OCR failed ({type(exc).__name__}).")
                    doc.pages.append(Page(page_number=index, extraction_method=method,
                                          has_usable_text=bool(text),
                                          extraction_error=None if text else "No usable text layer found."))
                    if text:
                        doc.pages[-1].paragraphs = _paragraphs_from_chunks(paragraph_texts or [text], index, method)
                    else:
                        doc.extraction_failures.append(f"Page {index}: no usable text layer.")
        except Exception as exc:
            doc.extraction_failures.append(f"PDF extraction failed ({type(exc).__name__}): {exc}")
            doc.exact_page_mapping = False
        return doc


def _paragraphs(text: str, page: int, method: str):
    from ai_text_audit.segmentation import make_paragraphs
    return make_paragraphs(text, page, method)


def _positioned_paragraphs(page) -> list[str]:
    """Use blank vertical gaps and first-line indentation as paragraph cues."""
    lines = []
    for word in page.extract_words(x_tolerance=2):
        if lines and abs(float(word["top"]) - float(lines[-1]["top"])) <= 2:
            line = lines[-1]
            line["words"].append(word["text"])
            line["bottom"] = max(line["bottom"], float(word["bottom"]))
        else:
            lines.append({"top": float(word["top"]), "bottom": float(word["bottom"]),
                          "x0": float(word["x0"]), "words": [word["text"]]})
    lines = [{**line, "text": " ".join(line["words"])} for line in lines
             if not (len(line["words"]) == 1 and line["words"][0].isdigit()
                     and float(line["top"]) > page.height * 0.85)]
    if not lines:
        return []
    body_lines = [line for line in lines if len(line["text"].strip()) >= 20]
    left_edge = min((float(line["x0"]) for line in body_lines), default=0.0)
    chunks: list[list[str]] = []
    current: list[str] = []
    previous = None
    for line in lines:
        text = line["text"].strip()
        height = max(float(line["bottom"]) - float(line["top"]), 1.0)
        vertical_gap = (float(line["top"]) - float(previous["bottom"])) if previous else 0.0
        indented = (len(text) >= 20 and float(line["x0"]) > left_edge + 8.0
                    and current and len(current[-1]) >= 25)
        if current and (vertical_gap > height * 0.65 or indented):
            chunks.append(current)
            current = []
        current.append(text)
        previous = line
    if current:
        chunks.append(current)
    return [" ".join(chunk) for chunk in chunks]


def _paragraphs_from_chunks(chunks: list[str], page: int, method: str):
    from ai_text_audit.models import Paragraph
    return [Paragraph(paragraph_id=f"p{page}-{number}", page_number=page,
                      paragraph_number=number, extracted_text=" ".join(chunk.split()),
                      extraction_method=method)
            for number, chunk in enumerate((item for item in chunks if item.strip()), start=1)]
