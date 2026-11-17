from pathlib import Path
import pytest
from ai_text_audit.extractors.tex import _latex_to_text


def test_tex_preserves_section_citation_equation_and_body():
    result = _latex_to_text(r"\section{A heading} Body text \cite{smith2020}. \begin{equation}x = y\end{equation}")
    assert "A heading" in result and "Body text" in result and "smith2020" in result and "x = y" in result


def test_pdf_page_extraction(tmp_path):
    from pypdf import PdfWriter
    from ai_text_audit.extractors.pdf import PDFExtractor
    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        writer.write(output)
    extracted = PDFExtractor().extract(path)
    assert len(extracted.pages) == 1
    assert extracted.pages[0].page_number == 1
    assert not extracted.pages[0].has_usable_text
    assert any("no usable text layer" in failure for failure in extracted.extraction_failures)


def test_tex_without_compiler_marks_page_mapping_unavailable(tmp_path, monkeypatch):
    from ai_text_audit.extractors import tex
    monkeypatch.setattr(tex.shutil, "which", lambda _: None)
    path = tmp_path / "sample.tex"
    path.write_text(r"\section{Intro}" + "\n" + "A sufficiently long body paragraph with useful content.", encoding="utf-8")
    doc = tex.TexExtractor().extract(path)
    assert not doc.exact_page_mapping
    assert doc.pages[0].page_number is None
    assert doc.pages[0].paragraphs[0].page_number is None
