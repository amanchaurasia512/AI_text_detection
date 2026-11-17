# AI Text Audit

A local Python command-line research pipeline for extracting paragraphs from PDF and LaTeX documents and recording configurable detector outputs. This project has no web server, dashboard, or rewriting feature.

## Install

Requires Python 3.11+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

PDF text extraction uses `pdfplumber`; PyMuPDF is used for OCR page rendering. Optional OCR requires the Python extra and the system Tesseract executable:

```bash
python -m pip install -e '.[ocr]'
# Install the Tesseract OCR binary using your operating system's package manager.
```

For exact LaTeX page mapping, install `latexmk` or `pdflatex`. Without either, the source parser extracts text and reports unavailable page mapping. The OCR flag enables fallback only for pages with no usable PDF text layer.

## Run

```bash
python -m ai_text_audit.cli analyze document.pdf --provider mock --output outputs/
python -m ai_text_audit.cli analyze document.tex --provider mock --output outputs/
python -m ai_text_audit.cli analyze data/samples/demo.tex --provider mock --output outputs/
```

For direct source checkout before installation, use `PYTHONPATH=src` with the same commands. Add `--ocr` to enable OCR fallback and `--verbose` for diagnostic logging. Reports are written to `outputs/report.json` and `outputs/report.md`.

The mock detector emits deterministic synthetic values strictly to exercise plumbing. It does not detect AI writing.

## Configure an API provider

No provider has been selected, so this repository deliberately contains no API URL, response schema, or claim of provider compatibility. The generic adapter sends a JSON object containing a configurable text field (default `text`). Select a provider, consult its documentation, then set `AUDIT_API_ENDPOINT`, `AUDIT_API_KEY`, and the appropriate `AUDIT_RESPONSE_*_PATH` dotted paths in `.env`. Configure `AUDIT_API_AUTH_HEADER` and `AUDIT_API_AUTH_PREFIX` to match its auth scheme. The adapter does not guess response fields. A score is stored as supplied and is not presumed to be calibrated probability. Leave response paths unset when no corresponding value is supplied.

```bash
cp .env.example .env
# Edit .env with the selected provider's documented values.
python -m ai_text_audit.cli analyze document.pdf --provider api
```

API keys are read from environment/config and are not deliberately included in reports or logs. Avoid mapping a secret-bearing response field.

## Fixtures and tests

The three text fixtures and `tests/test_cases/manifest.json` describe intended synthetic origins. They are pipeline examples, not ground-truth labels or detector validation. Run `pytest` to exercise extraction, segmentation, missing configuration, retry/error handling, mock processing, and report writing without a paid API key.

See [docs/architecture.md](docs/architecture.md) for data flow and limitations.
