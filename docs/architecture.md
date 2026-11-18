# Architecture

The CLI dispatches a validated `.pdf` or `.tex` path through four explicit stages: extraction, deterministic content classification, conservative paragraph reconstruction, and analysis eligibility. Extracted `SourceBlock` records are retained unchanged; output paragraphs carry source block IDs, page provenance, category evidence, processing status, and any reconstruction reason. The 80-character default is configurable with `AUDIT_MIN_PARAGRAPH_CHARS` and applies only to prose. Short prose and uncertain blocks remain in a manual-review queue.

## PDF extraction and page mapping

PDF extraction is page-wise through pdfplumber. It uses PDF content-stream order to reduce left/right column interleaving, removes ACL review line numbers, and records block geometry, line positions, font cues, table rules, and vector regions. A separate deterministic classifier distinguishes prose, headings, captions, table content, diagram/chart labels, references, footnotes, equations, headers/footers, and unknown items. Geometry and typography are evidence, not proof; uncertain blocks are retained for review. Raster image pages are noted; `--ocr` optionally tries Tesseract using PyMuPDF-rendered page images only when a page has no usable text layer. OCR dependencies and the Tesseract executable are optional. OCR failures are recorded.

The reconstruction pass joins only clear unterminated continuations (including line-end hyphenation and selected page/column transitions), and can split only where explicit layout evidence supplies a boundary. It preserves original source blocks and logs every join, split, or unresolved review decision. It does not infer boundaries from length alone. Table cell text remains flattened; cell relationships are not reconstructed.

## LaTeX compilation

The extractor tries `latexmk`, then `pdflatex` when latexmk is unavailable. It runs in the source project's directory and directs all compiler output to a temporary directory, leaving the paper source and its existing build results untouched. The inherited process environment is preserved; the source project's recursive path is prepended to `TEXINPUTS`, `BSTINPUTS`, and `BIBINPUTS`, retaining any caller-provided paths. This resolves nested inputs, styles, bibliography styles, bibliography files, and generated source files. Compiler output is captured as bytes and decoded with replacement for invalid UTF-8 so non-UTF-8 diagnostics cannot crash extraction. Nonzero exits, missing PDFs, and timeouts are reported as compile failures; page mapping is only marked exact when a generated PDF is successfully extracted.

If no compiler is available or compilation fails, a source text pass is used as a clearly marked partial fallback. It cannot produce page numbers or fully interpret arbitrary LaTeX macros and is never reported as successful compiled extraction.

## Detection adapter

`BaseDetector` provides a small synchronous interface. `MockDetector` returns repeatable synthetic values. `APIDetector` posts one paragraph per request through httpx. The endpoint, auth header, request text field, timeout, retry count, and response JSON paths are configurable. The provider has not been chosen; no endpoint or provider response schema is assumed. Transport errors expose only exception type in logs and report status. Rate limits and server failures are retried with bounded delay.

## Reports and interpretation

Reports expose both source-block and reconstructed-paragraph totals, category and processing-status counts, source provenance, page mappings, and reconstruction decisions. Only prose above the configured threshold reaches the detector. Short prose and uncertain classifications are marked for manual review; captions, table content, headings, references, and chart/diagram labels remain visible without prose detection. Page means use successful numeric scores only; they are descriptive. AI-like flagged counts require explicit AI-like labels and exclude uncertainty labels. No document-wide score is calculated because cross-paragraph aggregation depends on provider semantics. Detector outputs are indicators, never proof of authorship. Mock outputs are synthetic and do not assess authorship.

The fixture corpus tests the plumbing only. It does not measure accuracy, calibration, fairness, or robustness. API-provider integration requires a selected provider's actual documentation and a configured endpoint/field mapping.

## Benchmark milestones: P05 and P06

The research benchmark sequence is P01 → P02 → P03 → P04 → P05 → P06. P05 (`ollama_benchmark`) is the completed sequential baseline. P06 (`ollama_concurrency_benchmark`) is a separate experiment with configurable bounded concurrency (initial levels 2 and 3); it reads the same saved P04 report, applies the same prose/eligibility filter, length ordering, complete paragraph text, Ollama detector, and token metadata. It does not change P05 code or outputs.

P06 collects futures into stable input order and accounts for each eligible paragraph once. Resource sampling records system RAM, CPU and swap, identifiable Ollama process RSS/CPU, and optional NVIDIA GPU utilization/memory. Ollama process RSS is not total model memory. When the configured available-memory floor is crossed, P06 stops scheduling new work and lets in-flight requests finish within their detector timeouts; the run is marked stopped and remaining paragraphs are recorded as not run. Compare individual request latency, total workload completion time, and aggregate successful throughput separately against P05 under matched conditions. No performance improvement is claimed in advance.
