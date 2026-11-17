from pathlib import Path
import json
import pytest
from ai_text_audit.detectors.api import APIDetector
from ai_text_audit.detectors.mock import MockDetector


def test_mock_detector_is_repeatable():
    detector = MockDetector()
    assert detector.analyze("same paragraph").score == detector.analyze("same paragraph").score


def test_api_requires_explicit_endpoint(monkeypatch):
    monkeypatch.delenv("AUDIT_API_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="AUDIT_API_ENDPOINT"):
        APIDetector()


def test_api_rate_limit_retry_and_configured_mapping(monkeypatch):
    import httpx
    request = httpx.Request("POST", "https://example.invalid")
    replies = [httpx.Response(429, headers={"Retry-After": "0"}, request=request),
               httpx.Response(200, json={"result": {"value": 0.7, "label": "review"}}, request=request)]
    sent = []
    def mock_post(*args, **kwargs):
        sent.append(kwargs["json"])
        return replies.pop(0)
    monkeypatch.setattr(httpx, "post", mock_post)
    detector = APIDetector("https://example.invalid", retries=1, text_field="input.paragraph",
                           response_mapping={"score": "result.value", "classification": "result.label"})
    result = detector.analyze("sufficient test paragraph")
    assert result.score == 0.7 and result.classification == "review"
    assert sent[0] == {"input": {"paragraph": "sufficient test paragraph"}}


def test_api_failure_hides_transport_details(monkeypatch):
    import httpx
    def fail(*args, **kwargs):
        raise httpx.ConnectError("secret-like detail")
    monkeypatch.setattr(httpx, "post", fail)
    result = APIDetector("https://example.invalid", retries=0, api_key="topsecret").analyze("text")
    assert result.status == "failed"
    assert "topsecret" not in json.dumps(result.model_dump())
    assert "secret-like detail" not in (result.error or "")


def test_api_works_without_key_when_provider_auth_is_not_configured(monkeypatch):
    import httpx
    captured = {}
    def mock_post(*args, **kwargs):
        captured.update(kwargs)
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", args[0]))
    monkeypatch.setattr(httpx, "post", mock_post)
    result = APIDetector("https://example.invalid", api_key=None, retries=0).analyze("text")
    assert result.status == "success"
    assert "Authorization" not in captured["headers"]


def test_api_missing_key_with_configured_auth_fails_without_request(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("request should not be sent"))
    result = APIDetector("https://example.invalid", api_key=None, auth_header="Authorization").analyze("text")
    assert result.status == "failed"
    assert result.error == "API key is missing for the configured authentication header."
