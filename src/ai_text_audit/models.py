"""Typed document and detection records used throughout the pipeline."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    score: float | None = None
    classification: str | None = None
    confidence: float | None = None
    uncertainty: str | float | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    error: str | None = None


class Paragraph(BaseModel):
    paragraph_id: str
    page_number: int | None
    paragraph_number: int
    extracted_text: str
    extraction_method: str
    detection_result: DetectionResult = Field(default_factory=DetectionResult)


class Page(BaseModel):
    page_number: int | None
    paragraphs: list[Paragraph] = Field(default_factory=list)
    extraction_method: str = "text"
    extraction_error: str | None = None
    has_usable_text: bool = True


class Document(BaseModel):
    filename: str
    document_type: str
    pages: list[Page] = Field(default_factory=list)
    extraction_failures: list[str] = Field(default_factory=list)
    compilation_status: str | None = None
    exact_page_mapping: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentReport(BaseModel):
    document: Document
    provider: str
    generated_at: str
    limitations: list[str] = Field(default_factory=list)
    document_summary: dict[str, Any] = Field(default_factory=dict)
    page_summaries: list[dict[str, Any]] = Field(default_factory=list)
