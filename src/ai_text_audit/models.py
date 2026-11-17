"""Typed document and detection records used throughout the pipeline."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, model_validator


class DetectionResult(BaseModel):
    score: float | None = None
    classification: str | None = None
    confidence: float | None = None
    uncertainty: str | float | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    error: str | None = None


class SourceBlock(BaseModel):
    """An extraction unit retained as provenance for reconstructed paragraphs."""
    source_block_id: str
    page_number: int | None
    text: str
    extraction_method: str
    legacy_content_type: str = "paragraph"
    category: str = "unknown"
    classification_confidence: str = "low"
    classification_reason: str = "Not classified."
    bbox: dict[str, float] = Field(default_factory=dict)
    layout_evidence: dict[str, Any] = Field(default_factory=dict)


class Paragraph(BaseModel):
    paragraph_id: str
    page_number: int | None
    paragraph_number: int
    extracted_text: str
    extraction_method: str
    content_type: str = "paragraph"
    category: str = "prose"
    classification_confidence: str = "medium"
    classification_reason: str = "Default prose classification from extracted text block."
    source_block_ids: list[str] = Field(default_factory=list)
    source_page_numbers: list[int] = Field(default_factory=list)
    processing_status: str = "pending"
    reconstruction_status: str = "unchanged"
    reconstruction_reason: str | None = None
    detection_result: DetectionResult = Field(default_factory=DetectionResult)

    @model_validator(mode="after")
    def map_legacy_content_type(self) -> "Paragraph":
        if self.category != "prose":
            return self
        legacy_map = {"heading": "heading", "title": "heading", "caption": "figure_caption",
                      "table_row": "table_content", "graphic_label": "unknown", "author": "unknown"}
        if self.content_type in legacy_map:
            self.category = legacy_map[self.content_type]
            self.classification_confidence = "low"
            self.classification_reason = "Mapped from legacy extraction content type."
        return self


class Page(BaseModel):
    page_number: int | None
    paragraphs: list[Paragraph] = Field(default_factory=list)
    extraction_method: str = "text"
    extraction_error: str | None = None
    has_usable_text: bool = True
    content_notes: list[str] = Field(default_factory=list)


class Document(BaseModel):
    filename: str
    document_type: str
    pages: list[Page] = Field(default_factory=list)
    source_blocks: list[SourceBlock] = Field(default_factory=list)
    reconstruction_decisions: list[dict[str, Any]] = Field(default_factory=list)
    extraction_failures: list[str] = Field(default_factory=list)
    compilation_status: str | None = None
    extraction_status: str = "unknown"
    exact_page_mapping: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentReport(BaseModel):
    document: Document
    provider: str
    generated_at: str
    limitations: list[str] = Field(default_factory=list)
    document_summary: dict[str, Any] = Field(default_factory=dict)
    page_summaries: list[dict[str, Any]] = Field(default_factory=list)
