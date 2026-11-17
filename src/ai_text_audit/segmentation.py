import re
from ai_text_audit.models import Paragraph


def make_paragraphs(text: str, page_number: int | None, method: str) -> list[Paragraph]:
    chunks = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text)]
    return [Paragraph(paragraph_id=f"p{page_number or 'u'}-{i}", page_number=page_number,
                      paragraph_number=i, extracted_text=chunk, extraction_method=method)
            for i, chunk in enumerate((c for c in chunks if c), start=1)]
