from abc import ABC, abstractmethod
from ai_text_audit.models import DetectionResult


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def analyze(self, text: str) -> DetectionResult: ...
