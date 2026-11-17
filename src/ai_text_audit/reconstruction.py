"""Conservative, reversible paragraph reconstruction over retained PDF blocks."""
from __future__ import annotations

import re
from collections import defaultdict

from ai_text_audit.models import Document, Paragraph, SourceBlock

_SAFE_INTERVENING = {
    "heading", "figure_caption", "table_caption", "table_content",
    "diagram_label", "chart_label", "reference", "footnote", "header_footer",
    "equation",
}


def reconstruct_document(document: Document) -> None:
    """Rebuild clear prose continuations and retain every source block unchanged."""
    source_by_id = {block.source_block_id: block for block in document.source_blocks}
    original = [paragraph for page in document.pages for paragraph in page.paragraphs]
    expanded: list[Paragraph] = []

    # Split only where a producer supplied explicit visual paragraph-break offsets.
    for paragraph in original:
        block = source_by_id.get(paragraph.paragraph_id)
        offsets = (block.layout_evidence.get("paragraph_break_offsets", []) if block else [])
        pieces = _split_at_offsets(paragraph.extracted_text, offsets)
        if len(pieces) == 1:
            expanded.append(paragraph)
            continue
        document.reconstruction_decisions.append({
            "action": "split", "source_block_ids": [paragraph.paragraph_id],
            "page_numbers": paragraph.source_page_numbers or [paragraph.page_number],
            "reason": "Explicit layout evidence marked a paragraph boundary within the source block.",
            "automatic": True,
        })
        for number, text in enumerate(pieces, start=1):
            expanded.append(paragraph.model_copy(update={
                "paragraph_id": f"split-{paragraph.paragraph_id}-{number}",
                "extracted_text": text,
                "paragraph_number": paragraph.paragraph_number,
                "source_block_ids": [paragraph.paragraph_id],
                "reconstruction_status": "reconstructed_split",
                "reconstruction_reason": "Explicit layout paragraph-break offset.",
            }))

    out: list[Paragraph] = []
    index = 0
    while index < len(expanded):
        current = expanded[index]
        if current.category != "prose":
            out.append(current)
            index += 1
            continue

        next_index = index + 1
        intervening: list[Paragraph] = []
        while next_index < len(expanded) and expanded[next_index].category != "prose":
            candidate = expanded[next_index]
            if candidate.category not in _SAFE_INTERVENING:
                break
            intervening.append(candidate)
            next_index += 1

        if (next_index < len(expanded)
                and expanded[next_index].category == "prose"
                and _pages_are_adjacent(current, expanded[next_index])
                    and _continuation_reason(
                        current.extracted_text, expanded[next_index].extracted_text,
                        across_page=current.page_number != expanded[next_index].page_number)):
            joined = [current]
            reason_parts = []
            cursor = next_index
            saved_intervening = list(intervening)
            while cursor < len(expanded) and expanded[cursor].category == "prose":
                reason = _continuation_reason(
                    joined[-1].extracted_text, expanded[cursor].extracted_text,
                    across_page=joined[-1].page_number != expanded[cursor].page_number)
                if not reason or not _pages_are_adjacent(joined[-1], expanded[cursor]):
                    break
                joined.append(expanded[cursor])
                reason_parts.append(reason)
                cursor += 1
                while cursor < len(expanded) and expanded[cursor].category in _SAFE_INTERVENING:
                    saved_intervening.append(expanded[cursor])
                    cursor += 1
            merged = _merge_paragraphs(joined, reason_parts)
            out.append(merged)
            document.reconstruction_decisions.append({
                "action": "join", "source_block_ids": merged.source_block_ids,
                "page_numbers": merged.source_page_numbers,
                    "reason": reason or "Adjacent source blocks form one sentence.",
                "automatic": True,
            })
            # Retain skipped-over non-prose blocks, after the reconstructed prose unit.
            for block in saved_intervening:
                out.append(block)
            index = cursor
            continue

        if _looks_like_unresolved_continuation(current.extracted_text):
            current.processing_status = "reconstruction_review"
            current.reconstruction_status = "review"
            current.reconstruction_reason = "Text ends mid-sentence, but no safe continuation block was identified."
            document.reconstruction_decisions.append({
                "action": "review", "source_block_ids": current.source_block_ids,
                "page_numbers": current.source_page_numbers,
                "reason": current.reconstruction_reason, "automatic": False,
            })
        out.append(current)
        index += 1

    by_page: dict[int | None, list[Paragraph]] = defaultdict(list)
    for paragraph in out:
        by_page[paragraph.page_number].append(paragraph)
    for page in document.pages:
        page.paragraphs = by_page.get(page.page_number, [])
        for number, paragraph in enumerate(page.paragraphs, start=1):
            paragraph.paragraph_number = number
    document.metadata["source_block_count"] = len(document.source_blocks)
    document.metadata["reconstructed_paragraph_count"] = sum(len(p.paragraphs) for p in document.pages)


def _continuation_reason(left: str, right: str, *, across_page: bool = False) -> str | None:
    left = left.rstrip()
    right = right.lstrip()
    if not left or not right:
        return None
    if re.search(r"[\w\u00c0-\u024f]-$", left) and re.match(r"^[a-z\u00e0-\u024f]", right):
        return "Line-end hyphen plus lowercase continuation indicates a split word."
    if across_page and not re.search(r"[.!?][\"')\]]?$", left) and re.match(r"^[−–—,;:0-9]", right):
        return "Unterminated paragraph at a page edge continues with punctuation or a numerical expression."
    if not re.search(r"[.!?][\"')\]]?$", left) and re.match(r"^[a-z\u00e0-\u024f]", right):
        return "Unterminated source block followed by lowercase text indicates sentence continuation."
    return None


def _pages_are_adjacent(left: Paragraph, right: Paragraph) -> bool:
    pages = left.source_page_numbers or ([left.page_number] if left.page_number is not None else [])
    next_pages = right.source_page_numbers or ([right.page_number] if right.page_number is not None else [])
    return not pages or not next_pages or min(next_pages) - max(pages) <= 1


def _merge_paragraphs(paragraphs: list[Paragraph], reasons: list[str]) -> Paragraph:
    text = paragraphs[0].extracted_text
    for paragraph in paragraphs[1:]:
        right = paragraph.extracted_text.lstrip()
        if re.search(r"[\w\u00c0-\u024f]-$", text) and re.match(r"^[a-z\u00e0-\u024f]", right):
            text = text[:-1] + right
        else:
            text = f"{text.rstrip()} {right}"
    source_ids = [source_id for paragraph in paragraphs for source_id in
                  (paragraph.source_block_ids or [paragraph.paragraph_id])]
    source_pages = sorted({page for paragraph in paragraphs for page in
                           (paragraph.source_page_numbers or ([paragraph.page_number] if paragraph.page_number is not None else []))})
    first = paragraphs[0]
    return first.model_copy(update={
        "paragraph_id": "reconstructed-" + "-".join(source_ids),
        "extracted_text": text,
        "source_block_ids": source_ids,
        "source_page_numbers": source_pages,
        "reconstruction_status": "reconstructed_join",
        "reconstruction_reason": "; ".join(dict.fromkeys(reasons)),
    })


def _split_at_offsets(text: str, offsets: list[int]) -> list[str]:
    valid = sorted({int(offset) for offset in offsets if 0 < int(offset) < len(text)})
    if not valid:
        return [text]
    parts = []
    start = 0
    for offset in valid:
        left, right = text[start:offset].strip(), text[offset:].strip()
        if left and right and left[-1] in ".!?" and right[0].isupper():
            parts.append(left)
            start = offset
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _looks_like_unresolved_continuation(text: str) -> bool:
    return bool(re.search(r"(?:\w-?|\b(?:and|or|but|because|the|of|to|with|that))$", text.rstrip(), re.I)
                and not re.search(r"[.!?][\"')\]]?$", text.rstrip()))
