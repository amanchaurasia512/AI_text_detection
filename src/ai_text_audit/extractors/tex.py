"""LaTeX extractor; uses compiled PDF mapping when a compiler is available."""
from pathlib import Path
import logging
import shutil
import subprocess
import tempfile

from ai_text_audit.models import Document, Page
from .base import BaseExtractor

log = logging.getLogger(__name__)


class TexExtractor(BaseExtractor):
    def extract(self, path: Path, *, ocr: bool = False) -> Document:
        compiler = shutil.which("latexmk") or shutil.which("pdflatex")
        if compiler:
            with tempfile.TemporaryDirectory(prefix="ai-text-audit-") as temp:
                command = ([compiler, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            f"-outdir={temp}", path.name]
                           if Path(compiler).name == "latexmk" else
                           [compiler, "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={temp}", path.name])
                try:
                    result = subprocess.run(command, cwd=path.parent, capture_output=True, text=True, timeout=120)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    result = None
                    failure = type(exc).__name__
                pdf = Path(temp) / f"{path.stem}.pdf"
                if result is not None and result.returncode == 0 and pdf.exists():
                    from ai_text_audit.extractors.pdf import PDFExtractor
                    doc = PDFExtractor().extract(pdf, ocr=ocr)
                    doc.filename = path.name
                    doc.document_type = "tex"
                    doc.compilation_status = "success"
                    doc.metadata["compiler"] = Path(compiler).name
                    return doc
                failure = failure if result is None else (result.stderr or result.stdout)[-2000:]
                doc = _source_document(path)
                doc.compilation_status = "failed"
                doc.exact_page_mapping = False
                doc.extraction_failures.append(f"LaTeX compilation failed; exact PDF page mapping unavailable. {failure}")
                doc.metadata["compiler"] = Path(compiler).name
                return doc
        doc = _source_document(path)
        doc.compilation_status = "unavailable"
        doc.exact_page_mapping = False
        doc.extraction_failures.append("No pdflatex or latexmk found; exact PDF page mapping unavailable.")
        return doc


def _source_document(path: Path) -> Document:
    from ai_text_audit.segmentation import make_paragraphs
    source = _expand_inputs(path, set())
    text = _latex_to_text(source)
    return Document(filename=path.name, document_type="tex", exact_page_mapping=False,
                    pages=[Page(page_number=None, paragraphs=make_paragraphs(text, None, "tex-source"),
                                extraction_method="tex-source", has_usable_text=bool(text))])


def _expand_inputs(path: Path, seen: set[Path]) -> str:
    import re
    resolved = path.resolve()
    if resolved in seen:
        return ""
    seen.add(resolved)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    def replace(match):
        included = path.parent / match.group(1)
        if included.suffix == "":
            included = included.with_suffix(".tex")
        return "\n" + _expand_inputs(included, seen) + "\n"

    return re.sub(r"\\(?:input|include)\{([^{}]+)\}", replace, source)


def _latex_to_text(source: str) -> str:
    import re
    source = re.sub(r"(?<!\\)%.*", "", source)
    source = re.sub(r"\\begin\{(equation\*?|align\*?|gather\*?|verbatim|lstlisting)\}(.*?)\\end\{\1\}",
                    lambda m: "\n" + m.group(2).strip() + "\n", source, flags=re.S)
    source = re.sub(r"\\(?:section|subsection|subsubsection|chapter|paragraph)\*?\{([^{}]*)\}", r"\n\1\n", source)
    source = re.sub(r"\\(?:cite\w*|ref|pageref|label|url|bibliography|bibliographystyle)\*?(?:\[[^]]*\])?\{([^{}]*)\}", r"\1", source)
    source = re.sub(r"\\(?:textbf|textit|emph|underline|mathrm|mathbf|operatorname|title|author)\*?\{([^{}]*)\}", r"\1", source)
    source = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "\n", source)
    source = re.sub(r"\\(?:documentclass|usepackage|input|include|maketitle|tableofcontents|newcommand|renewcommand|setlength|hline|\[\]?)\b(?:\[[^]]*\])?\{[^{}]*\}", " ", source)
    source = re.sub(r"\\[a-zA-Z]+\*?", " ", source)
    source = source.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_").replace("~", " ")
    source = re.sub(r"[{}]", "", source)
    return re.sub(r"[ \t]+", " ", source).strip()
