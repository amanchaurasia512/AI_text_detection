"""Provider-neutral HTTP adapter. Provider paths and headers are explicit config."""
from __future__ import annotations
import json
import logging
import os
import time
from typing import Any

import httpx
from ai_text_audit.models import DetectionResult
from .base import BaseDetector

log = logging.getLogger(__name__)


def _path(data: Any, dotted: str | None) -> Any:
    if not dotted:
        return None
    for part in dotted.split("."):
        if isinstance(data, dict) and part in data:
            data = data[part]
        else:
            return None
    return data


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Place text at a configured dotted JSON request path."""
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError("Request text field must be a non-empty dotted path.")
    for part in parts[:-1]:
        child = data.get(part)
        if not isinstance(child, dict):
            child = {}
            data[part] = child
        data = child
    data[parts[-1]] = value


class APIDetector(BaseDetector):
    name = "api"

    def __init__(self, endpoint: str | None = None, *, api_key: str | None = None,
                 timeout: float = 30, retries: int = 2, text_field: str = "text",
                 response_mapping: dict[str, str] | None = None,
                 auth_header: str | None = None, auth_prefix: str = "Bearer "):
        self.endpoint = endpoint or os.getenv("AUDIT_API_ENDPOINT")
        if not self.endpoint:
            raise ValueError("AUDIT_API_ENDPOINT is required for --provider api.")
        self.api_key = api_key if api_key is not None else os.getenv("AUDIT_API_KEY")
        self.timeout, self.retries, self.text_field = timeout, retries, text_field
        self.mapping = response_mapping or {
            key: os.getenv(f"AUDIT_RESPONSE_{key.upper()}_PATH", "")
            for key in ("score", "classification", "confidence", "uncertainty")
        }
        self.auth_header = auth_header or os.getenv("AUDIT_API_AUTH_HEADER", "")
        self.auth_prefix = auth_prefix

    def analyze(self, text: str) -> DetectionResult:
        if self.auth_header and not self.api_key:
            return DetectionResult(status="failed", error="API key is missing for the configured authentication header.")
        headers = {"Content-Type": "application/json"}
        if self.auth_header and self.api_key:
            headers[self.auth_header] = f"{self.auth_prefix}{self.api_key}"
        body: dict[str, Any] = {}
        _set_path(body, self.text_field, text)
        attempts = max(0, self.retries) + 1
        for attempt in range(attempts):
            try:
                response = httpx.post(self.endpoint, json=body, headers=headers, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 < attempts:
                        delay = _retry_delay(response.headers.get("Retry-After"), attempt)
                        time.sleep(delay)
                        continue
                response.raise_for_status()
                payload = response.json()
                score = _path(payload, self.mapping.get("score"))
                return DetectionResult(
                    score=float(score) if score is not None else None,
                    classification=_as_str(_path(payload, self.mapping.get("classification"))),
                    confidence=_as_float(_path(payload, self.mapping.get("confidence"))),
                    uncertainty=_path(payload, self.mapping.get("uncertainty")),
                    provider_metadata={"provider_response": _redact(payload, self.api_key)}, status="success")
            except Exception as exc:
                if attempt + 1 >= attempts:
                    # Never log request headers, request body, or exception text from transport.
                    log.error("Detection API request failed (%s)", type(exc).__name__)
                    return DetectionResult(status="failed", error=f"API request failed ({type(exc).__name__}).")
                time.sleep(min(2 ** attempt, 8))
        return DetectionResult(status="failed", error="API request failed.")


def _retry_delay(value: str | None, attempt: int) -> float:
    try:
        return min(30.0, max(0.0, float(value))) if value else min(2 ** attempt, 8)
    except ValueError:
        return min(2 ** attempt, 8)


def _as_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _redact(value: Any, secret: str | None) -> Any:
    """Keep provider metadata useful while removing an echoed configured key."""
    if isinstance(value, dict):
        return {str(k): _redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[REDACTED]")
    return value
