"""PDF text extraction in content order with page-aware block segmentation."""
from __future__ import annotations

from pathlib import Path
import logging
import re
import statistics
from typing import Any

import pdfplumber

from ai_text_audit.models import Document, Page, Paragraph, SourceBlock
from .base import BaseExtractor

log = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    def extract(self, path: Path, *, ocr: bool = False) -> Document:
        document = Document(filename=path.name, document_type="pdf")
        try:
            with pdfplumber.open(path) as pdf:
                for page_number, pdf_page in enumerate(pdf.pages, start=1):
                    lines = _content_lines(pdf_page)
                    blocks = _segment_lines(lines, pdf_page.height)
                    page_text = "\n".join(block["text"] for block in blocks).strip()
                    method = "pdfplumber-content-order"
                    notes: list[str] = []

                    if pdf_page.images:
                        notes.append(
                            f"Page contains {len(pdf_page.images)} raster image(s); text embedded only in images "
                            "may not appear in extraction."
                        )

                    if not page_text and ocr:
                        page_text, ocr_error = _ocr_page(path, page_number)
                        if page_text:
                            method = "ocr:tesseract"
                            blocks = [{"text": chunk.strip(), "content_type": "paragraph"}
                                      for chunk in page_text.split("\n\n") if chunk.strip()]
                        elif ocr_error:
                            document.extraction_failures.append(
                                f"Page {page_number}: {ocr_error}"
                            )
                    page = Page(page_number=page_number, extraction_method=method,
                                has_usable_text=bool(page_text),
                                extraction_error=None if page_text else "No usable text layer found.",
                                content_notes=notes)
                    if blocks:
                        page.paragraphs = []
                        for number, block in enumerate(blocks, start=1):
                            block_id = f"p{page_number}-{number}"
                            paragraph = Paragraph(
                                paragraph_id=block_id, page_number=page_number,
                                paragraph_number=number, extracted_text=block["text"],
                                extraction_method=method,
                                content_type=block.get("content_type", "paragraph"),
                                source_block_ids=[block_id], source_page_numbers=[page_number],
                            )
                            page.paragraphs.append(paragraph)
                            document.source_blocks.append(SourceBlock(
                                source_block_id=block_id, page_number=page_number,
                                text=block["text"], extraction_method=method,
                                legacy_content_type=block.get("content_type", "paragraph"),
                                bbox=block.get("bbox", {}),
                                layout_evidence={**block.get("layout_evidence", {}),
                                                 "page_height": float(pdf_page.height),
                                                 "page_width": float(pdf_page.width)},
                            ))
                    else:
                        document.extraction_failures.append(
                            f"Page {page_number}: no usable text layer."
                        )
                    document.pages.append(page)
        except Exception as exc:
            document.extraction_failures.append(
                f"PDF extraction failed ({type(exc).__name__}): {exc}"
            )
            document.exact_page_mapping = False

        if not document.pages:
            document.extraction_status = "failed"
        elif all(page.has_usable_text for page in document.pages):
            document.extraction_status = "success"
        else:
            document.extraction_status = "partial"
        return document


def _ocr_page(path: Path, page_number: int) -> tuple[str, str | None]:
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        with fitz.open(path) as rendered_pdf:
            image = rendered_pdf[page_number - 1].get_pixmap(
                matrix=fitz.Matrix(2, 2), alpha=False
            )
        return pytesseract.image_to_string(
            Image.open(io.BytesIO(image.tobytes("png")))
        ).strip(), None
    except Exception as exc:
        log.warning("OCR failed on page %d (%s)", page_number, type(exc).__name__)
        return "", f"OCR failed ({type(exc).__name__})."


def _content_lines(page: Any) -> list[dict[str, Any]]:
    """Rebuild visual lines in the PDF's content-stream order.

    Content-stream order is materially better for multi-column papers than a
    simple top-to-bottom sort, which interleaves the left and right columns.
    """
    words = page.extract_words(
        use_text_flow=True, x_tolerance=1, extra_attrs=["fontname", "size"]
    )
    curves = getattr(page, "curves", [])
    try:
        table_bboxes = [table.bbox for table in page.find_tables()]
    except Exception:
        table_bboxes = []
    chart_boxes = [
        (float(rect["x0"]), float(rect["top"]), float(rect["x1"]), float(rect["bottom"]))
        for rect in getattr(page, "rects", [])
        if float(rect["x1"]) - float(rect["x0"]) > 40
        and float(rect["bottom"]) - float(rect["top"]) > 35
    ]
    diagram_boxes = [
        (float(curve["x0"]), float(curve["top"]), float(curve["x1"]), float(curve["bottom"]))
        for curve in curves
        if float(curve["x1"]) - float(curve["x0"]) > 20
        and float(curve["bottom"]) - float(curve["top"]) > 15
    ] if not chart_boxes else []
    lines: list[dict[str, Any]] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        top = float(word["top"])
        if _is_review_line_number(text, float(word["x0"]), top, page.width, page.height):
            continue
        if lines and abs(top - lines[-1]["top"]) <= 2.0:
            line = lines[-1]
            line["words"].append(word)
            line["bottom"] = max(line["bottom"], float(word["bottom"]))
        else:
            lines.append({"top": top, "bottom": float(word["bottom"]),
                          "x0": float(word["x0"]), "words": [word]})

    table_bands = _table_rule_bands(page, lines)
    for line in lines:
        line["text"] = " ".join(str(word["text"]) for word in line["words"])
        line["font_size"] = statistics.median(float(word.get("size", 10)) for word in line["words"])
        line["bold_ratio"] = sum(
            bool(re.search(r"bold|med|demi|semibold", str(word.get("fontname", "")), re.I))
            for word in line["words"]
        ) / len(line["words"])
        line["content_type"] = _classify_line(line)
        center_x = (min(float(w["x0"]) for w in line["words"]) +
                    max(float(w["x1"]) for w in line["words"])) / 2
        center_y = (float(line["top"]) + float(line["bottom"])) / 2
        short_label = len(line["text"]) <= 80
        line["graphic_kind"] = (
            "chart" if len(line["text"]) <= 35 and any(
                x0 - 45 <= center_x <= x1 + 45 and top - 18 <= center_y <= bottom + 75
                for x0, top, x1, bottom in chart_boxes
            )
            else "diagram" if short_label and any(
                x0 <= center_x <= x1 and top <= center_y <= bottom
                for x0, top, x1, bottom in diagram_boxes
            )
            else None
        )
        if line["graphic_kind"] and len(line["text"]) <= 110:
            line["content_type"] = "graphic_label"
        if line["content_type"] == "paragraph" and _looks_like_graphic_label(line):
            line["content_type"] = "graphic_label"
        if line["content_type"] == "paragraph" and (table_bboxes or table_bands):
            center_y = (line["top"] + line["bottom"]) / 2
            word_centers = [((float(word["x0"]) + float(word["x1"])) / 2)
                            for word in line["words"]]
            in_table_bbox = any(top <= center_y <= bottom and all(left <= x <= right for x in word_centers)
                                for left, top, right, bottom in table_bboxes)
            in_table_band = any(top <= center_y <= bottom and left <= min(word_centers)
                                and max(word_centers) <= right
                                for left, top, right, bottom in table_bands)
            if in_table_bbox or in_table_band:
                line["content_type"] = "table_row"
            line["table_region_nearby"] = in_table_band or any(
                left <= min(word_centers) and max(word_centers) <= right
                and (0 <= top - center_y <= 24 or 0 <= center_y - bottom <= 24)
                for left, top, right, bottom in table_bands
            )
        else:
            line["table_region_nearby"] = False
    return lines


def _table_rule_bands(page: Any, text_lines: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    """Infer table rectangles from repeated horizontal rules and intervening text."""
    grouped: dict[tuple[int, int], list[float]] = {}
    for line in getattr(page, "lines", []):
        if abs(float(line["bottom"]) - float(line["top"])) > 1.2:
            continue
        if float(line["x1"]) - float(line["x0"]) < 80:
            continue
        key = (round(float(line["x0"]) / 4), round(float(line["x1"]) / 4))
        grouped.setdefault(key, []).append(float(line["top"]))
    bands: list[tuple[float, float, float, float]] = []
    for (left_bucket, right_bucket), tops in grouped.items():
        left, right = left_bucket * 4.0, right_bucket * 4.0
        ordered = sorted(set(tops))
        for top, bottom in zip(ordered, ordered[1:]):
            if bottom - top < 14:
                continue
            enclosed = [line for line in text_lines if top <= line["top"] <= bottom
                        and line["x0"] >= left - 4 and line["x0"] <= right]
            if len(enclosed) >= 2:
                bands.append((left, top, right, bottom))
    return bands


def _is_review_line_number(text: str, x0: float, top: float,
                           width: float, height: float) -> bool:
    # ACL review-mode source line numbers sit in both outer gutters.
    if text.isdigit() and len(text) <= 4 and (x0 < width * 0.075 or x0 > width * 0.92):
        return True
    # A lone centered or outer-margin page number is not body content.
    if text.isdigit() and len(text) <= 3 and top > height * 0.88:
        return True
    return False


def _classify_line(line: dict[str, Any]) -> str:
    text = line["text"].strip()
    if text.lower().startswith("anonymous ") and line["top"] < 220:
        return "author"
    if (re.match(r"^(?:Figure|Fig\.|Table|Algorithm)\s*\d+\s*:", text, re.I)
            and line["font_size"] <= 10.2):
        return "caption"
    if _looks_like_graphic_label(line):
        return "graphic_label"
    if re.match(r"^[•●▪]\s+", text):
        return "list_item"
    # Section labels from article PDFs are generally short and set in a larger
    # or medium/bold font. Keep them as text blocks, but identify them as heads.
    if len(text) <= 110 and (line["font_size"] >= 11.5 or line["bold_ratio"] >= 0.95):
        if line["top"] < 180 and line["font_size"] >= 13:
            return "title"
        return "heading"
    return "paragraph"


def _looks_like_graphic_label(line: dict[str, Any]) -> bool:
    text = line["text"].strip()
    compact = re.sub(r"\s+", "", text)
    return (len(text) <= 2
            or bool(re.fullmatch(r"[−+\-]?\d+(?:[.,]\d+)*(?:[−+\-]?\d+(?:[.,]\d+)*)*", compact))
            or bool(re.fullmatch(r"[A-Z]{1,2}/[A-Za-z0-9_-]+", compact)))


def _segment_lines(lines: list[dict[str, Any]], page_height: float) -> list[dict[str, str]]:
    if not lines:
        return []
    line_starts = [line["x0"] for line in lines if len(line["text"]) >= 35
                   and line["content_type"] == "paragraph"]
    # Common line starts model each column's left edge, preserving first-line
    # indentation cues even on two-column pages.
    starts: dict[float, int] = {}
    for x in line_starts:
        bucket = round(x / 5) * 5
        starts[bucket] = starts.get(bucket, 0) + 1
    base_starts = [x for x, count in starts.items() if count >= 2]
    if not base_starts and starts:
        base_starts = [max(starts, key=starts.get)]

    blocks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None

    def flush() -> None:
        if not current:
            return
        text = _join_lines(current)
        if text:
            block_type = current[0]["content_type"]
            blocks.append(_block_metadata(text, block_type, current))
        current.clear()

    for line in lines:
        kind = line["content_type"]
        text = line["text"].strip()
        if kind in {"heading", "title", "author", "caption", "table_row", "graphic_label"}:
            flush()
            blocks.append(_block_metadata(text, kind, [line]))
            previous = line
            continue

        if kind == "list_item":
            flush()
            current.append(line)
            previous = line
            continue

        typical_height = max(float(line["bottom"]) - float(line["top"]), 1.0)
        gap = float(line["top"]) - float(previous["bottom"]) if previous else 0.0
        changed_column = bool(previous and float(line["top"]) < float(previous["top"]) - 3.0)
        if current and current[0]["content_type"] == "list_item":
            if changed_column or gap > typical_height * 0.9:
                flush()
            current.append(line)
            previous = line
            continue
        nearest_start = min(base_starts, key=lambda x: abs(x - float(line["x0"]))) if base_starts else float(line["x0"])
        indented_start = (bool(current) and len(current[-1]["text"]) >= 35
                          and float(line["x0"]) > nearest_start + 8.0)
        if current and (changed_column or gap > typical_height * 0.65 or indented_start):
            flush()
        current.append(line)
        previous = line
    flush()
    return _merge_caption_continuations(blocks)


def _block_metadata(text: str, content_type: str, lines: list[dict[str, Any]]) -> dict[str, Any]:
    words = [word for line in lines for word in line["words"]]
    return {
        "text": text,
        "content_type": content_type,
        "bbox": {
            "x0": min(float(word["x0"]) for word in words),
            "top": min(float(line["top"]) for line in lines),
            "x1": max(float(word["x1"]) for word in words),
            "bottom": max(float(line["bottom"]) for line in lines),
        },
        "layout_evidence": {
            "font_size_median": statistics.median(float(word.get("size", 10)) for word in words),
            "bold_ratio": sum(line["bold_ratio"] for line in lines) / len(lines),
            "graphic_kind": next((line.get("graphic_kind") for line in lines if line.get("graphic_kind")), None),
            "line_texts": [line["text"] for line in lines],
            "line_x0": [float(line["x0"]) for line in lines],
            "line_tops": [float(line["top"]) for line in lines],
            "line_bottoms": [float(line["bottom"]) for line in lines],
            "ends_with_hyphen": bool(re.search(r"[\w\u00c0-\u024f]-$", text)),
            "table_region_nearby": any(line.get("table_region_nearby") for line in lines),
        },
    }


def _merge_caption_continuations(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for block in blocks:
        if (block["content_type"] == "paragraph" and merged
                and merged[-1]["content_type"] == "caption"
                and not re.search(r"[.!?][\"')\]]?$", merged[-1]["text"])):
            merged[-1]["text"] = _join_text_fragments(merged[-1]["text"], block["text"])
            merged[-1]["bbox"]["x1"] = max(merged[-1]["bbox"]["x1"], block["bbox"]["x1"])
            merged[-1]["bbox"]["bottom"] = max(merged[-1]["bbox"]["bottom"], block["bbox"]["bottom"])
            merged[-1]["layout_evidence"]["line_texts"].extend(block["layout_evidence"]["line_texts"])
            merged[-1]["layout_evidence"]["line_x0"].extend(block["layout_evidence"]["line_x0"])
            merged[-1]["layout_evidence"]["line_tops"].extend(block["layout_evidence"]["line_tops"])
            merged[-1]["layout_evidence"]["line_bottoms"].extend(block["layout_evidence"]["line_bottoms"])
        else:
            merged.append(dict(block))
    return merged


def _join_text_fragments(left: str, right: str) -> str:
    text = f"{left} {right}"
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def _join_lines(lines: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for line in lines:
        text = line["text"].strip()
        parts.append(text)
    text = " ".join(parts)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text
