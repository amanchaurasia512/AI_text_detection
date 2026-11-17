
"""Local Ollama detector for experimental writing-style assessment."""
from __future__ import annotations

import json
import logging
import os
import asyncio
from contextlib import suppress
from typing import Any

import httpx

from ai_text_audit.models import DetectionResult
from .base import BaseDetector

log = logging.getLogger(__name__)


class OllamaDetector(BaseDetector):
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.getenv("AUDIT_OLLAMA_MODEL", "qwen2.5:3b")
        self.host = (host or os.getenv(
            "AUDIT_OLLAMA_HOST", "http://localhost:11434"
        )).rstrip("/")
        self.timeout = timeout

    def analyze(self, text: str) -> DetectionResult:
        try:
            response = httpx.post(
                f"{self.host}/api/generate", json=self._request_body(text), timeout=self.timeout,
            )
            response.raise_for_status()
            return self._decode_payload(response.json())
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            return self._failure(exc)

    def analyze_abortable(self, text: str, abort_event: Any) -> DetectionResult:
        """Run through this detector while allowing a benchmark safety event to cancel HTTP."""
        async def run() -> DetectionResult:
            async def request() -> DetectionResult:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(f"{self.host}/api/generate", json=self._request_body(text))
                    response.raise_for_status()
                    return self._decode_payload(response.json())

            task = asyncio.create_task(request())
            while not task.done():
                if abort_event.is_set():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    return DetectionResult(
                        status="failed",
                        error="Ollama request cancelled after critical memory threshold was reached.",
                        provider_metadata={"model": self.model, "runtime": "ollama", "aborted": True},
                    )
                await asyncio.sleep(0.1)
            try:
                return task.result()
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                return self._failure(exc)

        return asyncio.run(run())

    def _request_body(self, text: str) -> dict[str, Any]:
        prompt = f"""
You are an experimental writing-style analysis assistant.

Assess the writing characteristics of the supplied paragraph.
You cannot reliably determine whether a human or AI wrote text.
Do not treat polished grammar, formal language, or repetitive
structure as proof of AI authorship.

Return ONLY valid JSON with this structure:
{{
  "classification": "AI-like | human-like | uncertain",
  "evidence": ["observable writing characteristic"],
  "reasoning": "brief explanation",
  "limitations": "why this assessment is uncertain"
}}

Do not provide a numerical probability or confidence score.
If evidence is insufficient, use "uncertain".

PARAGRAPH:
{text}
"""
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }

    def _decode_payload(self, payload: dict[str, Any]) -> DetectionResult:
        raw = payload.get("response")
        if not isinstance(raw, str):
            raise ValueError("Ollama response field is missing.")
        result: dict[str, Any] = json.loads(raw)
        classification = result.get("classification")
        if classification not in {"AI-like", "human-like", "uncertain"}:
            raise ValueError("Invalid classification returned by model.")
        return DetectionResult(
            classification=classification,
            uncertainty="Experimental assessment; authorship is unverified.",
            status="success",
            provider_metadata={
                "model": payload.get("model", self.model), "runtime": "ollama",
                "evidence": result.get("evidence", []), "reasoning": result.get("reasoning", ""),
                "limitations": result.get("limitations", ""),
                "total_duration": payload.get("total_duration"),
                "load_duration": payload.get("load_duration"),
                "prompt_eval_count": payload.get("prompt_eval_count"),
                "prompt_eval_duration": payload.get("prompt_eval_duration"),
                "eval_count": payload.get("eval_count"),
                "eval_duration": payload.get("eval_duration"),
                "done_reason": payload.get("done_reason"),
            },
        )

    def _failure(self, exc: Exception) -> DetectionResult:
        log.error("Ollama analysis failed (%s)", type(exc).__name__)
        return DetectionResult(status="failed", error=f"Ollama analysis failed ({type(exc).__name__}).",
                               provider_metadata={"model": self.model, "runtime": "ollama"})
