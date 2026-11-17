"""Deterministic category classification kept separate from text extraction."""
from __future__ import annotations

import re
from collections import Counter

from ai_text_audit.models import Document, Paragraph, SourceBlock

CATEGORIES = {
    "prose", "heading", "figure_caption", "table_caption", "table_content",
    "diagram_label", "chart_label", "reference", "footnote", "equation",
    "header_footer", "unknown",
}


def classify_document(document: Document) -> None:
    """Classify retained source blocks using type, text, and PDF layout evidence."""
    repeated = Counter(
        _normalized(block.text)
        for block in document.source_blocks
        if block.text.strip() and _near_page_edge(block)
    )
    page_count = max(len(document.pages), 1)
    reference_section = False
    block_by_id: dict[str, SourceBlock] = {}
    for page in document.pages:
        for paragraph in page.paragraphs:
            block = next((b for b in document.source_blocks
                          if b.source_block_id == paragraph.paragraph_id), None)
            if block is None:
                block = SourceBlock(
                    source_block_id=paragraph.paragraph_id,
                    page_number=paragraph.page_number,
                    text=paragraph.extracted_text,
                    extraction_method=paragraph.extraction_method,
                    legacy_content_type=paragraph.content_type,
                    bbox=paragraph.model_extra.get("bbox", {}) if paragraph.model_extra else {},
                )
                document.source_blocks.append(block)
            block_by_id[block.source_block_id] = block
            text = block.text.strip()
            if re.match(r"^references\s*$", text, re.I) or re.match(r"^bibliography\s*$", text, re.I):
                reference_section = True
                category, confidence, reason = "heading", "high", "Bibliography section heading."
            elif re.match(r"^(?:appendix|[A-P](?:\s+|$))", text, re.I) and block.legacy_content_type == "heading":
                reference_section = False
                category, confidence, reason = _legacy_category(block, reference_section)
            else:
                category, confidence, reason = _classify_block(
                    block, reference_section=reference_section,
                    repeated=repeated[_normalized(text)] >= max(2, page_count // 2),
                )
            block.category = category
            block.classification_confidence = confidence
            block.classification_reason = reason
            paragraph.category = category
            paragraph.classification_confidence = confidence
            paragraph.classification_reason = reason
            if not paragraph.source_block_ids:
                paragraph.source_block_ids = [block.source_block_id]
            if paragraph.page_number is not None and not paragraph.source_page_numbers:
                paragraph.source_page_numbers = [paragraph.page_number]


def _classify_block(block: SourceBlock, *, reference_section: bool, repeated: bool) -> tuple[str, str, str]:
    text = block.text.strip()
    legacy = block.legacy_content_type
    if repeated:
        return "header_footer", "high", "Repeated identical text occurs at a page edge."
    if legacy in {"heading", "title"}:
        return "heading", "high", f"PDF block style classified as {legacy}."
    if legacy == "caption":
        if re.match(r"^Table\s*\d+\s*:", text, re.I):
            return "table_caption", "high", "Caption numbering begins with Table."
        if re.match(r"^(?:Figure|Fig\.)\s*\d+\s*:", text, re.I):
            return "figure_caption", "high", "Caption numbering begins with Figure/Fig."
        return "unknown", "low", "Caption block has no reliable figure/table marker."
    if legacy in {"table_row", "table_content"}:
        return "table_content", "high", "Text block lies within a detected PDF table region."
    if block.layout_evidence.get("table_region_nearby"):
        return "table_content", "medium", "Block is immediately adjacent to a rule-bounded table region."
    if legacy == "graphic_label":
        kind = block.layout_evidence.get("graphic_kind")
        if kind == "chart":
            return "chart_label", "high", "Block is within a dense chart/vector region."
        if kind == "diagram":
            return "diagram_label", "high", "Block is inside a drawn diagram node."
        return "unknown", "low", "Graphic label detected without reliable chart/diagram context."
    if legacy == "author":
        return "unknown", "low", "Author metadata is retained but has no dedicated P04 content category."
    if text.lower().startswith(("http://", "https://", "doi:")):
        return "footnote", "high", "URL/DOI text is retained as a reference note."
    if reference_section:
        return "reference", "high", "Block occurs in the bibliography region after the References heading."
    if _near_page_edge(block) and len(text) < 100 and _looks_like_reference(text):
        return "footnote", "medium", "Short citation-like text is positioned at a page edge."
    if _looks_like_equation(text):
        return "equation", "medium", "Compact mathematical notation detected."
    if legacy == "list_item":
        return "prose", "medium", "List item retained as analyzable text when it meets length requirements."
    if legacy in {"paragraph", "prose"}:
        return "prose", "medium", "Ordinary text block without stronger structural evidence."
    return "unknown", "low", f"No deterministic rule matched legacy type {legacy!r}."


def _legacy_category(block: SourceBlock, reference_section: bool) -> tuple[str, str, str]:
    return _classify_block(block, reference_section=reference_section, repeated=False)


def _looks_like_reference(text: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\b", text) and
                (re.search(r"\bet al\.?\b|\bIn Proceedings\b|\bPreprint\b|arXiv:|\bdoi\b", text, re.I)
                 or re.match(r"^[A-Z][^.!?]{2,100}\.\s+(?:19|20)\d{2}", text)))


def _looks_like_equation(text: str) -> bool:
    return len(text) < 180 and bool(re.search(r"(?:=|≤|≥|∑|∫|\b(?:min|max|argmax)\b).{0,80}(?:=|≤|≥|\d|[a-z])", text))


def _near_page_edge(block: SourceBlock) -> bool:
    y = block.bbox.get("top")
    height = block.layout_evidence.get("page_height")
    return bool(y is not None and height and (y < height * .07 or y > height * .90))


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
