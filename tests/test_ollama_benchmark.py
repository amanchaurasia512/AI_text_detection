import csv
import json
from pathlib import Path

from ai_text_audit.models import DetectionResult
from ai_text_audit import ollama_benchmark as benchmark


class RecordingDetector:
    name = "ollama"

    def __init__(self):
        self.inputs = []

    def analyze(self, text):
        self.inputs.append(text)
        return DetectionResult(status="success", classification="uncertain", provider_metadata={
            "model": "qwen2.5:3b", "prompt_eval_count": len(text) // 4,
            "prompt_eval_duration": 2_000_000, "eval_count": 12,
            "eval_duration": 4_000_000, "total_duration": 8_000_000,
            "load_duration": 1_000_000,
        })


def _input(path: Path, texts):
    path.write_text(json.dumps({"document": {"pages": [{"page_number": 1, "paragraphs": [
        {"paragraph_id": f"p{i}", "page_number": 1, "source_page_numbers": [1],
         "category": "prose", "processing_status": "eligible", "extracted_text": text}
        for i, text in enumerate(texts)
    ]}]}}), encoding="utf-8")


def test_load_chunks_sorts_by_length_and_retains_full_text(tmp_path):
    path = tmp_path / "p04.json"
    texts = ["z" * 1000, "a" * 72, "b" * 250]
    _input(path, texts)
    chunks = benchmark.load_eligible_chunks(path)
    assert [c["character_count"] for c in chunks] == [72, 250, 1000]
    assert [c["text"] for c in chunks] == [texts[1], texts[2], texts[0]]


def test_benchmark_sends_full_chunks_sequentially_and_records_token_rates(tmp_path, capsys):
    path = tmp_path / "p04.json"
    texts = ["x" * 900, "y" * 120]
    _input(path, texts)
    detector = RecordingDetector()
    output = tmp_path / "out"
    result = benchmark.run_benchmark(path, output, detector=detector, preflight=False,
                                    monitor_interval=0.1, minimum_available_gib=0.1)
    assert detector.inputs == [texts[1], texts[0]]
    assert result["run_status"] == "completed"
    assert result["successful_chunks"] == 2
    with (output / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["character_count"]) for row in rows] == [120, 900]
    assert all(row["output_tokens_per_second"] for row in rows)
    assert rows[1]["input_text_sha256"]
    assert "[1/2]" in capsys.readouterr().out


def test_preflight_failure_accounts_for_every_chunk_without_sending(tmp_path, monkeypatch):
    path = tmp_path / "p04.json"
    _input(path, ["first chunk", "second chunk"])
    detector = RecordingDetector()
    monkeypatch.setattr(benchmark, "check_ollama", lambda *_: (False, "server offline", {}))
    output = tmp_path / "out"
    result = benchmark.run_benchmark(path, output, detector=detector, preflight=True,
                                    monitor_interval=0.1, minimum_available_gib=0.1)
    assert result["run_status"] == "preflight_failed"
    assert result["not_run_chunks"] == 2
    assert detector.inputs == []
    with (output / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"not_run_preflight"}


def test_critical_memory_stops_before_any_chunk(tmp_path, monkeypatch):
    path = tmp_path / "p04.json"
    _input(path, ["chunk one", "chunk two"])
    detector = RecordingDetector()
    monkeypatch.setattr(benchmark, "_read_resource_sample", lambda _:
                        {"system_ram_total_bytes": 10 * benchmark.GiB,
                         "system_ram_available_bytes": 50 * 1024 ** 2,
                         "system_ram_used_bytes": 9 * benchmark.GiB,
                         "swap_total_bytes": 0, "swap_free_bytes": 0,
                         "ollama_process_rss_bytes": 0, "ollama_cpu_percent": None,
                         "system_cpu_percent": None, "gpu_utilization_percent": None,
                         "gpu_memory_used_mb": None, "gpu_memory_total_mb": None,
                         "swap_used_bytes": 0, "swap_in_pages": 0, "swap_out_pages": 0,
                         "timestamp_utc": "test"})
    output = tmp_path / "out"
    result = benchmark.run_benchmark(path, output, detector=detector, preflight=False,
                                    monitor_interval=0.1, minimum_available_gib=1.0)
    assert result["run_status"] == "stopped_low_memory"
    assert result["not_run_chunks"] == 2
    assert detector.inputs == []
