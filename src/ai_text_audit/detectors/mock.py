"""Deterministic pipeline stub; scores are illustrative only."""
from ai_text_audit.models import DetectionResult
from .base import BaseDetector


class MockDetector(BaseDetector):
    name = "mock"

    def analyze(self, text: str) -> DetectionResult:
        score = min(0.95, max(0.05, (sum(map(ord, text)) % 90) / 100))
        return DetectionResult(score=score, classification="mock-score", status="success",
                               provider_metadata={"note": "Synthetic score for pipeline testing; not a detector."})
