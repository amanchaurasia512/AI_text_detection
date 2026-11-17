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


def test_pdf_ocr_failure_is_recorded(tmp_path, monkeypatch):
    from pypdf import PdfWriter
    from ai_text_audit.extractors import pdf

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        writer.write(output)
    monkeypatch.setattr(pdf, "_ocr_page", lambda *_: ("", "OCR failed (ImportError)."))
    document = pdf.PDFExtractor().extract(path, ocr=True)
    assert document.extraction_status == "partial"
    assert any("OCR failed (ImportError)" in failure for failure in document.extraction_failures)


def test_tex_without_compiler_marks_page_mapping_unavailable(tmp_path, monkeypatch):
    from ai_text_audit.extractors import tex
    monkeypatch.setattr(tex.shutil, "which", lambda _: None)
    path = tmp_path / "sample.tex"
    path.write_text(r"\section{Intro}" + "\n" + "A sufficiently long body paragraph with useful content.", encoding="utf-8")
    doc = tex.TexExtractor().extract(path)
    assert not doc.exact_page_mapping
    assert doc.pages[0].page_number is None
    assert doc.pages[0].paragraphs[0].page_number is None


def test_tex_compiler_receives_environment_variables(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ai_text_audit.extractors import tex

    path = tmp_path / "main.tex"
    path.write_text(r"\section{Introduction} Test paragraph.", encoding="utf-8")

    monkeypatch.setattr(tex.shutil, "which", lambda _: "/usr/bin/latexmk")
    monkeypatch.setenv("TEXINPUTS", "/custom/acl//:")
    monkeypatch.setenv("BSTINPUTS", "/custom/acl//:")
    monkeypatch.setenv("BIBINPUTS", "/custom/bib:")

    captured = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        captured["kwargs_text"] = kwargs["text"]
        return SimpleNamespace(returncode=1, stderr="Simulated failure", stdout="")

    monkeypatch.setattr(tex.subprocess, "run", fake_run)
    tex.TexExtractor().extract(path)

    project_search = f"{tmp_path.resolve()}//:"
    assert captured["env"]["TEXINPUTS"] == project_search + "/custom/acl//:"
    assert captured["env"]["BSTINPUTS"] == project_search + "/custom/acl//:"
    assert captured["env"]["BIBINPUTS"] == project_search + "/custom/bib:"
    assert captured["kwargs_text"] is False


def test_tex_compiler_decodes_non_utf8_failure_diagnostics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ai_text_audit.extractors import tex

    path = tmp_path / "main.tex"
    path.write_text(r"\documentclass{article}\begin{document}Body\end{document}", encoding="utf-8")
    monkeypatch.setattr(tex.shutil, "which", lambda _: "/usr/bin/latexmk")

    def fake_run(command, **kwargs):
        assert kwargs["text"] is False
        return SimpleNamespace(returncode=1, stderr=b"Invalid byte: \xe9\nLaTeX Error", stdout=b"")

    monkeypatch.setattr(tex.subprocess, "run", fake_run)
    document = tex.TexExtractor().extract(path)
    assert document.compilation_status == "failed"
    assert "Invalid byte: \ufffd" in "\n".join(document.extraction_failures)
    assert "LaTeX Error" in "\n".join(document.extraction_failures)
    assert not document.exact_page_mapping


def test_tex_compiler_timeout_is_reported_with_partial_output(tmp_path, monkeypatch):
    import subprocess
    from ai_text_audit.extractors import tex

    path = tmp_path / "main.tex"
    path.write_text(r"\documentclass{article}\begin{document}Body\end{document}", encoding="utf-8")
    monkeypatch.setattr(tex.shutil, "which", lambda _: "/usr/bin/latexmk")

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=5, output=b"Before timeout \xe9", stderr=b"still working")

    monkeypatch.setattr(tex.subprocess, "run", fake_run)
    monkeypatch.setattr(tex.TexExtractor, "timeout_seconds", 5)
    document = tex.TexExtractor().extract(path)
    failure_text = "\n".join(document.extraction_failures)
    assert document.compilation_status == "failed"
    assert "timed out after 5 seconds" in failure_text
    assert "Before timeout \ufffd" in failure_text
    assert "still working" in failure_text


def test_tex_uses_pdflatex_fallback_when_latexmk_is_unavailable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ai_text_audit.extractors import tex

    path = tmp_path / "main.tex"
    path.write_text(r"\documentclass{article}\begin{document}Body\end{document}", encoding="utf-8")
    monkeypatch.setattr(tex.shutil, "which", lambda name: None if name == "latexmk" else "/usr/bin/pdflatex")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"fallback compiler failure")

    monkeypatch.setattr(tex.subprocess, "run", fake_run)
    document = tex.TexExtractor().extract(path)
    assert any(argument.startswith("-output-directory=") for argument in captured["command"])
    assert captured["cwd"] == path.parent.resolve()
    assert str(path.parent.resolve()) + "//:" in captured["env"]["TEXINPUTS"]
    assert document.compilation_status == "failed"


def test_tex_compiler_success_preserves_pdf_page_numbers(tmp_path):
    import shutil
    if not (shutil.which("latexmk") or shutil.which("pdflatex")):
        pytest.skip("LaTeX compiler is not installed")
    from ai_text_audit.extractors.tex import TexExtractor

    path = tmp_path / "mapped.tex"
    path.write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section{Background}\n"
        "This sufficiently long paragraph validates compiled PDF page mapping and provides enough text for mock analysis.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    document = TexExtractor().extract(path)
    assert document.compilation_status == "success"
    assert document.exact_page_mapping
    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1
    assert document.pages[0].paragraphs
    assert all("documentclass" not in p.extracted_text.lower()
               for page in document.pages for p in page.paragraphs)


def test_pdf_content_order_filters_review_line_numbers_and_keeps_columns():
    from ai_text_audit.extractors.pdf import _content_lines, _segment_lines

    def word(text, x0, top):
        return {"text": text, "x0": x0, "x1": x0 + len(text) * 5,
                "top": top, "bottom": top + 9, "size": 10,
                "fontname": "Nimbus-Regular"}

    class FakePage:
        width = 612
        height = 792

        def extract_words(self, **kwargs):
            return [word("Left", 80, 100), word("column", 105, 100),
                    word("001", 12, 101), word("continues", 80, 112),
                    word("Right", 330, 100), word("column", 355, 100),
                    word("049", 580, 101)]

    lines = _content_lines(FakePage())
    assert all("001" not in line["text"] and "049" not in line["text"] for line in lines)
    blocks = _segment_lines(lines, FakePage.height)
    content = [block["text"] for block in blocks]
    assert content == ["Left column continues", "Right column"]


def test_pdf_body_figure_reference_is_not_mislabeled_as_caption():
    from ai_text_audit.extractors.pdf import _classify_line

    body_reference = {"text": "Figure 1 summarises the pipeline.", "font_size": 10.91,
                      "bold_ratio": 0.0, "top": 200}
    actual_caption = {"text": "Figure 1: Pipeline overview.", "font_size": 9.96,
                      "bold_ratio": 0.0, "top": 500}
    assert _classify_line(body_reference) == "paragraph"
    assert _classify_line(actual_caption) == "caption"
