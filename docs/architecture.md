# Architecture

The CLI dispatches a validated `.pdf` or `.tex` path to an extractor, creates page and paragraph records, independently runs the chosen detector on each sufficiently long paragraph, and serializes a document report in JSON and Markdown.

## Extraction and page mapping

PDF extraction is page-wise through pdfplumber. Pages without usable text are retained and recorded as extraction failures; `--ocr` optionally tries Tesseract using PyMuPDF-rendered page images. OCR dependencies and the Tesseract executable are optional. Paragraph separation follows blank lines in extracted page text, so PDF layout can still affect boundaries.

For TeX, `latexmk` or `pdflatex` is attempted in a temporary directory. A successful compile is extracted as PDF, retaining physical page numbers. A failed or unavailable compiler triggers a source text pass for common headings, citations, formatting commands, equations, and environments; it sets page numbers to unavailable and records that exact PDF mapping is not available. Source parsing cannot fully interpret arbitrary macro definitions.

## Detection adapter

`BaseDetector` provides a small synchronous interface. `MockDetector` returns repeatable synthetic values. `APIDetector` posts one paragraph per request through httpx. The endpoint, auth header, text field, timeout, retry count, and response JSON paths are configurable. The provider has not been chosen; no endpoint or provider response schema is assumed. Transport errors expose only exception type in logs and report status. Rate limits and server failures are retried with bounded delay.

## Reports and interpretation

Paragraph results retain source text, page and paragraph identifiers, extraction method, score, classification, confidence/uncertainty, status, and error. Very short text is marked `insufficient_text` without a fabricated score. Page means are shown when numeric scores exist; they are descriptive only. No document-wide score is calculated because cross-paragraph aggregation depends on provider semantics. Detector outputs are indicators, never proof of authorship.

The fixture corpus tests the plumbing only. It does not measure accuracy, calibration, fairness, or robustness. API-provider integration requires a selected provider's actual documentation and a configured endpoint/field mapping.
