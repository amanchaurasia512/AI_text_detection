from abc import ABC, abstractmethod
from pathlib import Path
from ai_text_audit.models import Document


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, path: Path, *, ocr: bool = False) -> Document: ...
