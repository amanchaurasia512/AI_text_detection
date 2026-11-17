
import json
import httpx

from ai_text_audit.detectors.ollama import OllamaDetector


def test_ollama_valid_response(monkeypatch):
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "response": json.dumps({
                    "classification": "uncertain",
                    "evidence": ["Repetitive sentence structure"],
                    "reasoning": "Insufficient evidence.",
                    "limitations": "Authorship cannot be established."
                }),
                "total_duration": 1000000,
                "load_duration": 100000,
                "prompt_eval_count": 18,
                "prompt_eval_duration": 200000,
                "eval_count": 50,
                "eval_duration": 500000,
                "done_reason": "stop",
            },
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OllamaDetector().analyze("This is a sample paragraph for testing.")

    assert result.status == "success"
    assert result.classification == "uncertain"
    assert result.score is None
    assert result.provider_metadata["model"] == "qwen2.5:3b"
    assert result.provider_metadata["prompt_eval_count"] == 18
    assert result.provider_metadata["prompt_eval_duration"] == 200000
    assert result.provider_metadata["eval_count"] == 50
    assert result.provider_metadata["eval_duration"] == 500000
    assert result.provider_metadata["load_duration"] == 100000
    assert result.provider_metadata["done_reason"] == "stop"


def test_ollama_invalid_json(monkeypatch):
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            json={"response": "not valid json"},
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OllamaDetector().analyze("Sample paragraph.")

    assert result.status == "failed"
    assert result.error is not None


def test_ollama_invalid_classification(monkeypatch):
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "response": json.dumps({
                    "classification": "probably AI",
                    "evidence": [],
                    "reasoning": "",
                    "limitations": "",
                })
            },
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OllamaDetector().analyze("Sample paragraph.")

    assert result.status == "failed"


def test_ollama_service_unavailable(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OllamaDetector().analyze("Sample paragraph.")

    assert result.status == "failed"
    assert result.error is not None


def test_ollama_abortable_request_cancels_when_safety_event_is_set(monkeypatch):
    import asyncio
    import threading

    class SlowClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            await asyncio.sleep(5)

    monkeypatch.setattr(httpx, "AsyncClient", SlowClient)
    abort = threading.Event()
    timer = threading.Timer(0.15, abort.set)
    timer.start()
    result = OllamaDetector(timeout=10).analyze_abortable("paragraph", abort)
    timer.join()
    assert result.status == "failed"
    assert result.provider_metadata["aborted"] is True
