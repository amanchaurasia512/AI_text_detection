"""LaTeX extractor; uses compiled PDF mapping when a compiler is available."""
from pathlib import Path
import logging
import shutil
import subprocess
import tempfile
import os
from subprocess import TimeoutExpired

from ai_text_audit.models import Document, Page
from .base import BaseExtractor

log = logging.getLogger(__name__)


class TexExtractor(BaseExtractor):
    timeout_seconds = 180

    def extract(self, path: Path, *, ocr: bool = False) -> Document:
        path = path.resolve()
        compiler = shutil.which("latexmk") or shutil.which("pdflatex")
        if compiler:
            with tempfile.TemporaryDirectory(prefix="ai-text-audit-") as temp:
                command = ([compiler, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            f"-outdir={temp}", path.name]
                           if Path(compiler).name == "latexmk" else
                           [compiler, "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={temp}", path.name])
                environment = _compiler_environment(path.parent)
                try:
                    result = subprocess.run(command, cwd=path.parent, env=environment,
                                            capture_output=True, text=False,
                                            timeout=self.timeout_seconds)
                    returncode = result.returncode
                    diagnostics = _compiler_diagnostics(result.stdout, result.stderr)
                    failure_kind = None
                except TimeoutExpired as exc:
                    result = None
                    returncode = None
                    diagnostics = _compiler_diagnostics(exc.stdout, exc.stderr)
                    failure_kind = f"Compilation timed out after {self.timeout_seconds} seconds."
                except OSError as exc:
                    result = None
                    returncode = None
                    diagnostics = str(exc)
                    failure_kind = f"Compiler could not be started ({type(exc).__name__})."
                pdf = Path(temp) / f"{path.stem}.pdf"
                if returncode == 0 and pdf.is_file():
                    from ai_text_audit.extractors.pdf import PDFExtractor
                    doc = PDFExtractor().extract(pdf, ocr=ocr)
                    doc.filename = path.name
                    doc.document_type = "tex"
                    doc.compilation_status = "success"
                    doc.metadata["compiler_diagnostics"] = _summarize_diagnostics(diagnostics)
                    doc.metadata["compiler"] = Path(compiler).name
                    if not doc.pages:
                        doc.compilation_status = "failed"
                        doc.extraction_failures.append(
                            "Compiler returned success but its PDF could not be extracted."
                        )
                        doc.exact_page_mapping = False
                    return doc
                doc = _source_document(path)
                doc.compilation_status = "failed"
                doc.extraction_status = "partial" if doc.pages and doc.pages[0].paragraphs else "failed"
                doc.exact_page_mapping = False
                detail = failure_kind or f"Compiler exited with status {returncode}."
                if returncode == 0 and not pdf.is_file():
                    detail = "Compiler returned success but did not produce the expected PDF."
                if diagnostics:
                    detail += "\n" + _summarize_diagnostics(diagnostics)
                doc.extraction_failures.append(
                    "LaTeX compilation failed; exact PDF page mapping unavailable. " + detail
                )
                doc.metadata["compiler"] = Path(compiler).name
                return doc
        doc = _source_document(path)
        doc.compilation_status = "unavailable"
        doc.extraction_status = "partial" if doc.pages and doc.pages[0].paragraphs else "failed"
        doc.exact_page_mapping = False
        doc.extraction_failures.append("No pdflatex or latexmk found; exact PDF page mapping unavailable.")
        return doc


def _compiler_environment(project_root: Path) -> dict[str, str]:
    """Preserve the caller environment and add recursive project dependency paths."""
    environment = os.environ.copy()
    root = str(project_root.resolve())
    for variable in ("TEXINPUTS", "BSTINPUTS", "BIBINPUTS"):
        existing = environment.get(variable, "")
        project_path = f"{root}//"
        entries = existing.split(os.pathsep) if existing else []
        if project_path not in entries:
            environment[variable] = project_path + os.pathsep + existing
    return environment


def _decode_compiler_output(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _compiler_diagnostics(stdout: bytes | str | None, stderr: bytes | str | None) -> str:
    out = _decode_compiler_output(stdout).strip()
    err = _decode_compiler_output(stderr).strip()
    return "\n".join(part for part in (out, err) if part)[-8000:]


def _summarize_diagnostics(diagnostics: str) -> str:
    return diagnostics[-4000:]


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
