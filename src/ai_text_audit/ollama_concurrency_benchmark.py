"""Concurrent, resource-aware benchmark for local Ollama inference.

P06 deliberately shares P05's input contract, detector, result metadata and
resource sampler. Requests are bounded by a thread pool and results are emitted
in P05's stable length order, regardless of completion order.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_text_audit import ollama_benchmark as p05
from ai_text_audit.detectors.ollama import OllamaDetector


def _invoke(detector: Any, chunk: dict[str, Any]) -> tuple[Any | None, str, str | None, float]:
    started = time.perf_counter()
    try:
        # Pass the complete source string unchanged. The normal detector timeout
        # bounds each request; no benchmark retry is performed.
        result = detector.analyze(chunk["text"])
        return result, result.status, result.error, time.perf_counter() - started
    except Exception as exc:
        return None, "failed", f"Ollama call failed ({type(exc).__name__}).", time.perf_counter() - started


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = p05.load_eligible_chunks(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    provenance = {}
    for page in data["document"]["pages"]:
        for paragraph in page["paragraphs"]:
            if paragraph.get("category") == "prose" and paragraph.get("processing_status") == "eligible":
                provenance[paragraph["paragraph_id"]] = paragraph.get("source_block_ids", [])
    for chunk in chunks:
        chunk["source_block_ids"] = provenance.get(chunk["paragraph_id"], [])
    return chunks


def _result_row(chunk: dict[str, Any], result: Any | None, latency: float | None,
                samples: list[dict[str, Any]], status: str | None = None,
                error: str | None = None) -> dict[str, Any]:
    row = p05._chunk_row(chunk, result, latency, samples, status=status, error=error)
    row["source_block_ids"] = ";".join(chunk.get("source_block_ids", []))
    return row


def run_benchmark(input_path: Path, output_parent: Path, *, model: str = "qwen2.5:3b",
                  host: str = "http://localhost:11434", timeout: float = 180.0,
                  concurrency: int = 2, preflight: bool = True, monitor_interval: float = 0.5,
                  minimum_available_gib: float = 1.0, minimum_available_percent: float = 5.0,
                  detector: Any | None = None) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    chunks = _load_chunks(input_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_parent / f"ollama_concurrency_{stamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = output_parent / f"ollama_concurrency_{stamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    start_utc = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    detector = detector or OllamaDetector(model=model, host=host, timeout=timeout)
    preflight_ok, preflight_error, preflight_data = (p05.check_ollama(host, model)
        if preflight else (True, None, {"preflight": "disabled"}))
    monitor = p05.ResourceMonitor(interval=monitor_interval,
        minimum_available_bytes=int(minimum_available_gib * p05.GiB),
        minimum_available_percent=minimum_available_percent)
    monitor.start()
    initial = monitor.latest()
    stop_reason: str | None = None
    run_status = "completed"
    slots: list[dict[str, Any] | None] = [None] * len(chunks)

    if initial.get("critical_memory"):
        stop_reason = initial.get("critical_memory_reason", "Memory safety threshold reached before scheduling.")
        run_status = "stopped_low_memory"
    elif not preflight_ok:
        stop_reason = preflight_error or "Ollama preflight failed."
        run_status = "preflight_failed"
    else:
        next_index = 0
        running: dict[futures.Future, tuple[int, int, float]] = {}
        executor = futures.ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="ollama-p06")
        try:
            while running or (next_index < len(chunks) and run_status != "stopped_low_memory"):
                while next_index < len(chunks) and len(running) < concurrency:
                    sample = monitor.sample_once()
                    if monitor.critical_event.is_set() or sample.get("critical_memory"):
                        stop_reason = sample.get("critical_memory_reason", "Memory safety threshold reached; new requests stopped.")
                        run_status = "stopped_low_memory"
                        break
                    mark = monitor.mark()
                    index = next_index
                    started = time.perf_counter()
                    future = executor.submit(_invoke, detector, chunks[index])
                    running[future] = (index, mark, started)
                    next_index += 1
                if running:
                    done, _ = futures.wait(running, timeout=max(0.05, monitor_interval),
                                           return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        index, mark, _ = running.pop(future)
                        try:
                            result, status, error, latency = future.result()
                        except Exception as exc:  # protects accounting if wrapper itself fails
                            result, status, error, latency = None, "failed", f"Ollama call failed ({type(exc).__name__}).", 0.0
                        after = monitor.sample_once()
                        row = _result_row(chunks[index], result, latency,
                            monitor.samples_since(mark), status=status, error=error)
                        slots[index] = row
                        print(p05._progress_line(index + 1, len(chunks), chunks[index], row, after), flush=True)
                elif next_index < len(chunks) and run_status != "stopped_low_memory":
                    # Defensive guard against a non-progressing scheduler.
                    continue
            if run_status == "stopped_low_memory":
                for index in range(next_index, len(chunks)):
                    slots[index] = _result_row(chunks[index], None, None, [],
                        status="not_run_low_memory", error=stop_reason)
        finally:
            # In-flight requests finish normally, bounded by the detector timeout.
            executor.shutdown(wait=True, cancel_futures=False)
    if run_status in {"preflight_failed", "stopped_low_memory"}:
        for index, chunk in enumerate(chunks):
            if slots[index] is None:
                status = "not_run_preflight" if run_status == "preflight_failed" else "not_run_low_memory"
                slots[index] = _result_row(chunk, None, None, [], status=status, error=stop_reason)

    monitor.stop()
    rows = [row for row in slots if row is not None]
    runtime = time.perf_counter() - run_start
    samples = monitor.samples
    attempted = sum(r["status"] in {"success", "failed"} for r in rows)
    succeeded = sum(r["status"] == "success" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    not_run = len(rows) - attempted
    input_token_values = [r["input_tokens"] for r in rows if r.get("input_tokens") is not None]
    output_token_values = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    latency_values = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    summary = {
        "run_status": run_status, "start_time_utc": start_utc,
        "completed_time_utc": datetime.now(timezone.utc).isoformat(), "model": model, "host": host,
        "input_report": str(input_path), "output_directory": str(output_dir),
        "eligible_chunks": len(chunks), "chunks_attempted": attempted,
        "successful_chunks": succeeded, "failed_chunks": failed, "not_run_chunks": not_run,
        "configured_concurrency": concurrency, "total_runtime_seconds": runtime,
        "paragraphs_per_minute": succeeded / runtime * 60 if runtime > 0 else 0.0,
        "total_input_tokens": sum(input_token_values), "total_output_tokens": sum(output_token_values),
        "aggregate_output_tokens_per_second": sum(output_token_values) / runtime if runtime > 0 else 0.0,
        "mean_paragraph_latency_seconds": (sum(latency_values) / len(latency_values) if latency_values else None),
        "stop_reason": stop_reason,
        "preflight": {"ok": preflight_ok, "error": preflight_error, **preflight_data},
        "system_ram_total_mb": initial.get("system_ram_total_bytes", 0) / (1024 ** 2) if initial.get("system_ram_total_bytes") else None,
        "peak_system_ram_mb": p05._peak(samples, "system_ram_used_bytes") / (1024 ** 2) if p05._peak(samples, "system_ram_used_bytes") is not None else None,
        "minimum_system_ram_available_mb": min((s["system_ram_available_bytes"] / (1024 ** 2) for s in samples if s.get("system_ram_available_bytes") is not None), default=None),
        "peak_ollama_process_rss_mb": p05._peak(samples, "ollama_process_rss_bytes") / (1024 ** 2) if p05._peak(samples, "ollama_process_rss_bytes") is not None else None,
        "ollama_rss_measurement_note": "Aggregate RSS of identifiable Ollama processes; not total model memory.",
        "mean_ollama_cpu_percent": p05._mean(samples, "ollama_cpu_percent"),
        "mean_system_cpu_percent": p05._mean(samples, "system_cpu_percent"),
        "peak_gpu_utilization_percent": p05._peak(samples, "gpu_utilization_percent"),
        "peak_gpu_memory_used_mb": p05._peak(samples, "gpu_memory_used_mb"),
        "peak_swap_used_mb": p05._peak(samples, "swap_used_bytes") / (1024 ** 2) if p05._peak(samples, "swap_used_bytes") is not None else None,
        "resource_sample_count": len(samples), "long_chunks_truncated_or_discarded": 0,
        "benchmark_semantics": "Concurrent full-paragraph requests. Per-paragraph wall latency is distinct from total workload completion time and aggregate throughput. No performance improvement is presumed.",
    }
    result_fields = ["paragraph_id", "page_number", "source_page_numbers", "source_block_ids",
        "character_count", "input_text_sha256", "status", "error", "classification",
        "latency_seconds", "input_tokens", "output_tokens", "input_tokens_per_second",
        "output_tokens_per_second", "total_duration_ns", "load_duration_ns",
        "prompt_eval_duration_ns", "eval_duration_ns", "peak_system_ram_mb",
        "minimum_system_ram_available_mb", "peak_ollama_process_ram_mb", "mean_ollama_cpu_percent",
        "mean_system_cpu_percent", "peak_gpu_utilization_percent", "peak_gpu_memory_used_mb",
        "swap_used_mb", "swap_in_pages_delta", "swap_out_pages_delta"]
    p05._write_dict_rows(output_dir / "results.csv", rows, result_fields)
    groups = p05._length_summaries(rows)
    p05._write_dict_rows(output_dir / "length_range_summary.csv", groups)
    (output_dir / "resource_samples.csv").write_text("", encoding="utf-8")
    p05._write_dict_rows(output_dir / "resource_samples.csv", samples)
    (output_dir / "report.json").write_text(json.dumps({"summary": summary, "results": rows, "resource_samples": samples}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "report.md", summary, groups)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any], groups: list[dict[str, Any]]) -> None:
    lines = ["# P06 Concurrent Ollama Performance Benchmark", "",
        f"- Run status: **{summary['run_status']}**", f"- Model: `{summary['model']}`",
        f"- Configured concurrency: {summary['configured_concurrency']}",
        f"- Paragraphs total / attempted / succeeded / failed / not run: {summary['eligible_chunks']} / {summary['chunks_attempted']} / {summary['successful_chunks']} / {summary['failed_chunks']} / {summary['not_run_chunks']}",
        f"- Overall workload completion time: {summary['total_runtime_seconds']:.2f} seconds",
        f"- Aggregate successful throughput: {summary['paragraphs_per_minute']:.2f} paragraphs/minute",
        f"- Stop reason: {summary.get('stop_reason') or 'none'}", "",
        "Per-paragraph latency is recorded in `results.csv`; it measures each request's wall time and is not the same as overall completion time. Aggregate throughput is successful paragraphs divided by total run time. Compare these measures to P05 only for the same input report, model, host, and similar machine state. P05 is the sequential baseline; P06 is an experiment. Results alone determine whether concurrency helped.", "",
        "Ollama process RSS identifies server/runner process memory only; it is not total model memory. Resource samples can miss brief peaks.", "",
        "## Length ranges", "", "| Characters | Paragraphs | Success | Failed | Not run | Mean request latency (s) |", "|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        label = f"{group['min_chars']}–{group['max_chars']}" if group["max_chars"] is not None else f"{group['min_chars']}+"
        lines.append(f"| {label} | {group['count']} | {group['successes']} | {group['failures']} | {group['not_run']} | {p05._fmt(group['mean_latency_seconds'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run concurrent P06 local Ollama inference benchmark.")
    parser.add_argument("--input", type=Path, default=Path("outputs/p04_category_aware_20261117T115439Z/report.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--host", default=os.getenv("AUDIT_OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Maximum simultaneous requests (initial comparison levels: 2 and 3).")
    parser.add_argument("--low-memory-gib", type=float, default=1.0)
    parser.add_argument("--low-memory-percent", type=float, default=5.0)
    args = parser.parse_args()
    try:
        result = run_benchmark(args.input, args.output, model=args.model, host=args.host,
            timeout=args.timeout, concurrency=args.concurrency,
            minimum_available_gib=args.low_memory_gib, minimum_available_percent=args.low_memory_percent)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {type(exc).__name__}: {exc}\n")
    return 0 if result["run_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
