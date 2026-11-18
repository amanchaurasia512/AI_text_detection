import csv
import json
import threading
import time
from pathlib import Path

from ai_text_audit import ollama_benchmark as p05
from ai_text_audit import ollama_concurrency_benchmark as p06
from ai_text_audit.models import DetectionResult


def _input(path: Path, texts):
    path.write_text(json.dumps({"document": {"pages": [{"page_number": 1, "paragraphs": [
        {"paragraph_id": f"p{i}", "page_number": 1, "source_page_numbers": [1],
         "source_block_ids": [f"block-{i}"],
         "category": "prose", "processing_status": "eligible", "extracted_text": text}
        for i, text in enumerate(texts)
    ]}]}}), encoding="utf-8")


class FakeMonitor:
    stop_at_sample = None

    def __init__(self, **kwargs):
        self.samples = []
        self.critical_event = threading.Event()
        self.calls = 0

    def start(self):
        self.sample_once()

    def sample_once(self):
        self.calls += 1
        critical = self.stop_at_sample is not None and self.calls >= self.stop_at_sample
        if critical:
            self.critical_event.set()
        sample = {"timestamp_utc": "test", "system_ram_total_bytes": 1000,
                  "system_ram_available_bytes": 1 if critical else 999,
                  "system_ram_used_bytes": 1, "swap_total_bytes": 0, "swap_free_bytes": 0,
                  "swap_used_bytes": 0, "ollama_process_rss_bytes": 10,
                  "ollama_cpu_percent": 1, "system_cpu_percent": 1,
                  "gpu_utilization_percent": None, "gpu_memory_used_mb": None,
                  "gpu_memory_total_mb": None, "gpu_name": None,
                  "critical_memory": critical,
                  "critical_memory_reason": "test safety floor" if critical else None}
        self.samples.append(sample)
        return sample

    def mark(self):
        return len(self.samples)

    def samples_since(self, mark):
        return self.samples[mark:]

    def latest(self):
        return self.samples[-1]

    def stop(self):
        self.sample_once()


class DelayedDetector:
    def __init__(self):
        self.inputs = []
        self.lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def analyze(self, text):
        with self.lock:
            self.inputs.append(text)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.06 if text.startswith("slow") else 0.01)
        with self.lock:
            self.active -= 1
        if text.startswith("failure"):
            raise TimeoutError("simulated request timeout")
        return DetectionResult(status="success", classification="uncertain",
                               provider_metadata={"prompt_eval_count": 1, "eval_count": 2,
                                                 "eval_duration": 1_000_000})


def _run(monkeypatch, tmp_path, texts, detector, **kwargs):
    monkeypatch.setattr(p05, "ResourceMonitor", FakeMonitor)
    FakeMonitor.stop_at_sample = None
    source = tmp_path / "p04.json"
    _input(source, texts)
    result = p06.run_benchmark(source, tmp_path / "runs", detector=detector,
        preflight=False, monitor_interval=0.01, minimum_available_gib=0, **kwargs)
    return result, Path(result["output_directory"])


def test_bounded_scheduling_exactly_once_and_deterministic_order(monkeypatch, tmp_path):
    texts = ["slow" + "a" * 80, "fast" + "b" * 20, "middle" + "c" * 21]
    detector = DelayedDetector()
    result, output = _run(monkeypatch, tmp_path, texts, detector, concurrency=2)
    assert detector.maximum_active == 2
    assert sorted(detector.inputs) == sorted(texts)
    rows = list(csv.DictReader((output / "results.csv").open(encoding="utf-8-sig")))
    assert [row["paragraph_id"] for row in rows] == ["p1", "p2", "p0"]
    assert [row["source_block_ids"] for row in rows] == ["block-1", "block-2", "block-0"]
    assert result["chunks_attempted"] == result["eligible_chunks"] == 3
    assert result["run_status"] == "completed"


def test_failures_timeouts_and_full_long_input_are_accounted(monkeypatch, tmp_path):
    long_text = "failure " + "x" * 10000
    detector = DelayedDetector()
    result, output = _run(monkeypatch, tmp_path, [long_text, "ok" * 50], detector, concurrency=2)
    rows = list(csv.DictReader((output / "results.csv").open(encoding="utf-8-sig")))
    failed = next(row for row in rows if row["paragraph_id"] == "p0")
    assert failed["status"] == "failed"
    assert "TimeoutError" in failed["error"]
    assert long_text in detector.inputs
    assert result["failed_chunks"] == 1
    assert result["long_chunks_truncated_or_discarded"] == 0


def test_memory_stop_allows_inflight_to_finish_and_marks_remaining_not_run(monkeypatch, tmp_path):
    class SafetyMonitor(FakeMonitor):
        def sample_once(self):
            # start, then two admissions are healthy; next scheduling check trips.
            self.calls += 1
            critical = self.calls >= 4
            if critical:
                self.critical_event.set()
            sample = {"timestamp_utc": "test", "system_ram_total_bytes": 1000,
                      "system_ram_available_bytes": 1 if critical else 999,
                      "system_ram_used_bytes": 1, "swap_total_bytes": 0, "swap_free_bytes": 0,
                      "swap_used_bytes": 0, "ollama_process_rss_bytes": 10,
                      "ollama_cpu_percent": 1, "system_cpu_percent": 1,
                      "gpu_utilization_percent": None, "gpu_memory_used_mb": None,
                      "gpu_memory_total_mb": None, "gpu_name": None,
                      "critical_memory": critical, "critical_memory_reason": "test safety floor" if critical else None}
            self.samples.append(sample)
            return sample
    monkeypatch.setattr(p05, "ResourceMonitor", SafetyMonitor)
    source = tmp_path / "p04.json"
    _input(source, ["a" * 30, "b" * 31, "c" * 32, "d" * 33])
    detector = DelayedDetector()
    result = p06.run_benchmark(source, tmp_path / "runs", detector=detector, preflight=False,
        concurrency=2, minimum_available_gib=0, monitor_interval=0.01)
    rows = list(csv.DictReader((Path(result["output_directory"]) / "results.csv").open(encoding="utf-8-sig")))
    assert result["run_status"] == "stopped_low_memory"
    assert result["stop_reason"] == "test safety floor"
    assert result["successful_chunks"] == 2
    assert result["not_run_chunks"] == 2
    assert len(detector.inputs) == 2
    assert all(row["status"].startswith("not_run") for row in rows[2:])


def test_empty_and_preflight_partial_outputs(monkeypatch, tmp_path):
    detector = DelayedDetector()
    result, output = _run(monkeypatch, tmp_path, [], detector, concurrency=3)
    assert result["run_status"] == "completed"
    assert result["eligible_chunks"] == result["chunks_attempted"] == 0
    assert all((output / name).exists() for name in
               ("report.json", "report.md", "results.csv", "length_range_summary.csv", "resource_samples.csv"))

    source = tmp_path / "partial.json"
    _input(source, ["a" * 100, "b" * 110])
    monkeypatch.setattr(p05, "check_ollama", lambda *_: (False, "offline", {}))
    result = p06.run_benchmark(source, tmp_path / "failed", detector=detector, preflight=True,
                               concurrency=2, minimum_available_gib=0)
    assert result["run_status"] == "preflight_failed"
    assert result["not_run_chunks"] == 2
    rows = list(csv.DictReader((Path(result["output_directory"]) / "results.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 2
