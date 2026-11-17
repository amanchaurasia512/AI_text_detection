"""Sequential, resource-aware benchmark for the project's OllamaDetector."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ai_text_audit.detectors.ollama import OllamaDetector

GiB = 1024 ** 3


def load_eligible_chunks(path: Path) -> list[dict[str, Any]]:
    """Load every eligible prose unit without shortening or rewriting its text."""
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for page in data["document"]["pages"]:
        for paragraph in page["paragraphs"]:
            if paragraph.get("category") == "prose" and paragraph.get("processing_status") == "eligible":
                text = paragraph.get("extracted_text")
                if not isinstance(text, str):
                    raise ValueError(f"Paragraph {paragraph.get('paragraph_id')} has no text string.")
                chunks.append({
                    "paragraph_id": paragraph["paragraph_id"],
                    "page_number": paragraph.get("page_number"),
                    "source_page_numbers": paragraph.get("source_page_numbers", []),
                    "text": text,
                    "character_count": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                })
    # Stable ordering: length first, then original document traversal order for ties.
    return sorted(chunks, key=lambda chunk: chunk["character_count"])


def check_ollama(host: str, model: str, timeout: float = 5.0) -> tuple[bool, str | None, dict[str, Any]]:
    """Check API reachability and whether the requested model is already local."""
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout, trust_env=False)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        names = {m.get("name") for m in models if isinstance(m, dict)}
        if model not in names:
            return False, f"Requested model {model!r} is not present in local Ollama tags.", {"available_models": sorted(n for n in names if n)}
        return True, None, {"available_models": sorted(n for n in names if n)}
    except (httpx.HTTPError, ValueError) as exc:
        return False, f"Ollama preflight failed ({type(exc).__name__}).", {"exception_type": type(exc).__name__}


class ResourceMonitor:
    """Samples Linux RAM/CPU/swap and optional NVIDIA utilization while inference runs."""

    def __init__(self, interval: float = 0.5, minimum_available_bytes: int | None = None,
                 minimum_available_percent: float = 5.0):
        self.interval = max(0.1, interval)
        self.minimum_available_bytes = minimum_available_bytes or GiB
        self.minimum_available_percent = minimum_available_percent
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.critical_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self._last_system_cpu: tuple[int, int] | None = None
        self._last_process_cpu: tuple[float, int] | None = None
        self._last_gpu_query = 0.0
        self._last_live_print = 0.0
        self._last_gpu: dict[str, Any] = {"gpu_utilization_percent": None, "gpu_memory_used_mb": None,
                                          "gpu_memory_total_mb": None, "gpu_name": None}

    def start(self) -> None:
        self.sample_once()
        self.thread = threading.Thread(target=self._run, name="ollama-resource-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(2.0, self.interval * 4))
        self.sample_once()

    def mark(self) -> int:
        with self.lock:
            return len(self.samples)

    def samples_since(self, mark: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.samples[mark:])

    def latest(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.samples[-1]) if self.samples else {}

    def sample_once(self) -> dict[str, Any]:
        sample = _read_resource_sample(self)
        total = sample.get("system_ram_total_bytes") or 0
        available = sample.get("system_ram_available_bytes")
        if available is not None and total:
            critical_floor = max(self.minimum_available_bytes, int(total * self.minimum_available_percent / 100))
            if available < critical_floor:
                self.critical_event.set()
                sample["critical_memory"] = True
                sample["critical_memory_reason"] = (
                    f"Available system RAM {available / GiB:.2f} GiB fell below the "
                    f"{critical_floor / GiB:.2f} GiB safety floor."
                )
        swap_total = sample.get("swap_total_bytes") or 0
        swap_free = sample.get("swap_free_bytes")
        if swap_total and swap_free is not None and swap_free / swap_total < 0.05:
            self.critical_event.set()
            sample["critical_memory"] = True
            sample["critical_memory_reason"] = "Swap free space fell below 5% of configured swap."
        with self.lock:
            self.samples.append(sample)
        return sample

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            sample = self.sample_once()
            now = time.monotonic()
            if now - self._last_live_print >= 1.0:
                self._last_live_print = now
                print(_live_resource_line(sample), flush=True)


def _read_resource_sample(monitor: ResourceMonitor) -> dict[str, Any]:
    now = time.monotonic()
    mem = _read_meminfo()
    pids, rss, process_ticks = _ollama_process_usage()
    system_cpu = _read_system_cpu()
    system_cpu_percent = None
    if monitor._last_system_cpu and system_cpu:
        prev_total, prev_idle = monitor._last_system_cpu
        total, idle = system_cpu
        delta_total, delta_idle = total - prev_total, idle - prev_idle
        if delta_total > 0:
            system_cpu_percent = max(0.0, 100 * (delta_total - delta_idle) / delta_total)
    monitor._last_system_cpu = system_cpu

    process_cpu_percent = None
    if process_ticks is not None:
        if monitor._last_process_cpu:
            prev_time, prev_ticks = monitor._last_process_cpu
            elapsed = now - prev_time
            if elapsed > 0:
                process_cpu_percent = 100 * (process_ticks - prev_ticks) / os.sysconf("SC_CLK_TCK") / elapsed
        monitor._last_process_cpu = (now, process_ticks)

    if now - monitor._last_gpu_query >= 1.0:
        monitor._last_gpu = _read_nvidia_smi()
        monitor._last_gpu_query = now

    vm = _read_vmstat()
    sample = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system_ram_total_bytes": mem.get("MemTotal"),
        "system_ram_available_bytes": mem.get("MemAvailable"),
        "system_ram_used_bytes": (mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)) if mem else None,
        "ollama_process_ids": pids,
        "ollama_process_rss_bytes": rss if pids else None,
        "ollama_cpu_percent": process_cpu_percent,
        "system_cpu_percent": system_cpu_percent,
        "swap_total_bytes": mem.get("SwapTotal"),
        "swap_free_bytes": mem.get("SwapFree"),
        "swap_used_bytes": (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) if mem else None,
        "swap_in_pages": vm.get("pswpin"),
        "swap_out_pages": vm.get("pswpout"),
        **monitor._last_gpu,
    }
    return sample


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    return values


def _ollama_process_usage() -> tuple[list[int], int | None, int | None]:
    pids: list[int] = []
    rss_total = 0
    cpu_total = 0
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return pids, None, None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip().lower()
            if not comm.startswith("ollama"):
                continue
            status = (entry / "status").read_text()
            rss_line = next((line for line in status.splitlines() if line.startswith("VmRSS:")), None)
            rss_kb = int(rss_line.split()[1]) if rss_line else 0
            stat = (entry / "stat").read_text().split()
            cpu_ticks = int(stat[13]) + int(stat[14])
            pids.append(int(entry.name))
            rss_total += rss_kb * 1024
            cpu_total += cpu_ticks
        except (OSError, ValueError, IndexError):
            continue
    return pids, rss_total if pids else None, cpu_total if pids else None


def _read_system_cpu() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = [int(x) for x in fields]
        return sum(values), values[3] + (values[4] if len(values) > 4 else 0)
    except (OSError, ValueError, IndexError):
        return None


def _read_vmstat() -> dict[str, int]:
    try:
        return {key: int(value) for key, value in
                (line.split() for line in Path("/proc/vmstat").read_text().splitlines())
                if key in {"pswpin", "pswpout"}}
    except (OSError, ValueError):
        return {}


def _read_nvidia_smi() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1.5, check=True)
        rows = []
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 4:
                rows.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
        if rows:
            return {"gpu_name": "; ".join(r[0] for r in rows),
                    "gpu_utilization_percent": max(r[1] for r in rows),
                    "gpu_memory_used_mb": sum(r[2] for r in rows),
                    "gpu_memory_total_mb": sum(r[3] for r in rows)}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {"gpu_name": None, "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None, "gpu_memory_total_mb": None}


def _peak(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return max(values) if values else None


def _mean(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return statistics.mean(values) if values else None


def _chunk_row(chunk: dict[str, Any], result: Any | None, latency: float | None,
               samples: list[dict[str, Any]], status: str | None = None,
               error: str | None = None) -> dict[str, Any]:
    metadata = result.provider_metadata if result else {}
    prompt_tokens = metadata.get("prompt_eval_count")
    output_tokens = metadata.get("eval_count")
    prompt_ns = metadata.get("prompt_eval_duration")
    output_ns = metadata.get("eval_duration")
    return {
        "paragraph_id": chunk["paragraph_id"], "page_number": chunk["page_number"],
        "source_page_numbers": ";".join(map(str, chunk["source_page_numbers"])),
        "character_count": chunk["character_count"], "input_text_sha256": chunk["sha256"],
        "status": status or (result.status if result else "not_run"),
        "error": error or (result.error if result else None),
        "classification": result.classification if result else None,
        "latency_seconds": latency,
        "input_tokens": prompt_tokens, "output_tokens": output_tokens,
        "input_tokens_per_second": (prompt_tokens / (prompt_ns / 1e9) if prompt_tokens is not None and prompt_ns else None),
        "output_tokens_per_second": (output_tokens / (output_ns / 1e9) if output_tokens is not None and output_ns else None),
        "total_duration_ns": metadata.get("total_duration"),
        "load_duration_ns": metadata.get("load_duration"),
        "prompt_eval_duration_ns": prompt_ns, "eval_duration_ns": output_ns,
        "peak_system_ram_mb": (_peak(samples, "system_ram_used_bytes") / (1024 ** 2)
                                if _peak(samples, "system_ram_used_bytes") is not None else None),
        "minimum_system_ram_available_mb": (_peak([{**s, "inv": -s["system_ram_available_bytes"]}
                                                    for s in samples if s.get("system_ram_available_bytes") is not None], "inv") / -(1024 ** 2)
                                              if any(s.get("system_ram_available_bytes") is not None for s in samples) else None),
        "peak_ollama_process_ram_mb": (_peak(samples, "ollama_process_rss_bytes") / (1024 ** 2)
                                       if _peak(samples, "ollama_process_rss_bytes") is not None else None),
        "mean_ollama_cpu_percent": _mean(samples, "ollama_cpu_percent"),
        "mean_system_cpu_percent": _mean(samples, "system_cpu_percent"),
        "peak_gpu_utilization_percent": _peak(samples, "gpu_utilization_percent"),
        "peak_gpu_memory_used_mb": _peak(samples, "gpu_memory_used_mb"),
        "swap_used_mb": (_peak(samples, "swap_used_bytes") / (1024 ** 2)
                         if _peak(samples, "swap_used_bytes") is not None else None),
        "swap_in_pages_delta": _counter_delta(samples, "swap_in_pages"),
        "swap_out_pages_delta": _counter_delta(samples, "swap_out_pages"),
    }


def _counter_delta(samples: list[dict[str, Any]], key: str) -> int | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    return values[-1] - values[0] if len(values) > 1 else (0 if values else None)


def run_benchmark(input_path: Path, output_dir: Path, *, model: str = "qwen2.5:3b",
                  host: str = "http://localhost:11434", timeout: float = 180.0,
                  preflight: bool = True, monitor_interval: float = 0.5,
                  minimum_available_gib: float = 1.0,
                  minimum_available_percent: float = 5.0,
                  detector: OllamaDetector | None = None) -> dict[str, Any]:
    chunks = load_eligible_chunks(input_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    start_utc = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    detector = detector or OllamaDetector(model=model, host=host, timeout=timeout)
    preflight_ok, preflight_error, preflight_metadata = (check_ollama(host, model)
                                                          if preflight else (True, None, {"preflight": "disabled"}))
    monitor = ResourceMonitor(interval=monitor_interval,
                              minimum_available_bytes=int(minimum_available_gib * GiB),
                              minimum_available_percent=minimum_available_percent)
    monitor.start()
    initial = monitor.latest()
    low_memory_at_start = bool(initial.get("critical_memory"))

    rows: list[dict[str, Any]] = []
    run_status = "completed"
    stop_reason = None
    total = len(chunks)
    if low_memory_at_start:
        run_status = "stopped_low_memory"
        stop_reason = initial.get("critical_memory_reason", "Critical memory threshold reached before benchmark start.")
        print(f"STOP before first chunk: {stop_reason}", flush=True)
        rows.extend(_chunk_row(chunk, None, None, [], status="not_run_low_memory", error=stop_reason)
                    for chunk in chunks)
    elif not preflight_ok:
        run_status = "preflight_failed"
        stop_reason = preflight_error
        print(f"Preflight failed; no chunks sent: {preflight_error}", flush=True)
        for chunk in chunks:
            rows.append(_chunk_row(chunk, None, None, [], status="not_run_preflight", error=preflight_error))
    else:
        for index, chunk in enumerate(chunks, start=1):
            before = monitor.sample_once()
            if monitor.critical_event.is_set():
                run_status = "stopped_low_memory"
                stop_reason = before.get("critical_memory_reason", "Critical memory threshold reached before request.")
                print(f"\nSTOP: {stop_reason}", flush=True)
                rows.extend(_chunk_row(rest, None, None, [], status="not_run_low_memory", error=stop_reason)
                            for rest in chunks[index - 1:])
                break

            mark = monitor.mark()
            started = time.perf_counter()
            try:
                # Exact P04 text is passed to the existing detector; no truncation or cleanup occurs here.
                if hasattr(detector, "analyze_abortable"):
                    result = detector.analyze_abortable(chunk["text"], monitor.critical_event)
                else:
                    result = detector.analyze(chunk["text"])
                error = result.error
                status = result.status
            except Exception as exc:
                result = None
                error = f"Ollama call failed ({type(exc).__name__})."
                status = "failed"
            latency = time.perf_counter() - started
            after = monitor.sample_once()
            samples = monitor.samples_since(mark)
            row = _chunk_row(chunk, result, latency, samples, status=status, error=error)
            rows.append(row)
            print(_progress_line(index, total, chunk, row, after), flush=True)

            if monitor.critical_event.is_set():
                run_status = "stopped_low_memory"
                stop_reason = after.get("critical_memory_reason", "Critical memory threshold reached during request.")
                print(f"STOP before next chunk: {stop_reason}", flush=True)
                if index < total:
                    rows.extend(_chunk_row(rest, None, None, [], status="not_run_low_memory", error=stop_reason)
                                for rest in chunks[index:])
                break

    monitor.stop()
    total_runtime = time.perf_counter() - run_start
    resource_samples = monitor.samples
    successful = sum(row["status"] == "success" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    not_run = sum(row["status"].startswith("not_run") for row in rows)
    summary = {
        "run_status": run_status, "start_time_utc": start_utc,
        "completed_time_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "host": host, "input_report": str(input_path),
        "eligible_chunks": total, "chunks_attempted": successful + failed,
        "successful_chunks": successful, "failed_chunks": failed, "not_run_chunks": not_run,
        "total_runtime_seconds": total_runtime, "stop_reason": stop_reason,
        "preflight": {"ok": preflight_ok, "error": preflight_error, **preflight_metadata},
        "system_ram_total_mb": (initial.get("system_ram_total_bytes") / (1024 ** 2)
                                 if initial.get("system_ram_total_bytes") else None),
        "peak_system_ram_mb": (_peak(resource_samples, "system_ram_used_bytes") / (1024 ** 2)
                               if _peak(resource_samples, "system_ram_used_bytes") is not None else None),
        "minimum_system_ram_available_mb": min(
            (s["system_ram_available_bytes"] / (1024 ** 2) for s in resource_samples
             if s.get("system_ram_available_bytes") is not None), default=None),
        "peak_ollama_process_ram_mb": (_peak(resource_samples, "ollama_process_rss_bytes") / (1024 ** 2)
                                       if _peak(resource_samples, "ollama_process_rss_bytes") is not None else None),
        "mean_ollama_cpu_percent": _mean(resource_samples, "ollama_cpu_percent"),
        "mean_system_cpu_percent": _mean(resource_samples, "system_cpu_percent"),
        "peak_gpu_utilization_percent": _peak(resource_samples, "gpu_utilization_percent"),
        "peak_gpu_memory_used_mb": _peak(resource_samples, "gpu_memory_used_mb"),
        "peak_swap_used_mb": (_peak(resource_samples, "swap_used_bytes") / (1024 ** 2)
                              if _peak(resource_samples, "swap_used_bytes") is not None else None),
        "resource_sample_count": len(resource_samples),
        "long_chunks_truncated_or_discarded": 0,
        "benchmark_semantics": "Sequential OllamaDetector calls; wall latency includes the complete request/response. Token throughput uses Ollama prompt_eval/eval metadata when supplied.",
    }
    _write_results_csv(output_dir / "results.csv", rows)
    length_summaries = _length_summaries(rows)
    _write_dict_rows(output_dir / "length_range_summary.csv", length_summaries)
    (output_dir / "report.json").write_text(json.dumps({
        "summary": summary, "results": rows,
        "resource_samples": resource_samples,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "report.md", summary, rows, length_summaries)
    return summary


def _progress_line(index: int, total: int, chunk: dict[str, Any], row: dict[str, Any],
                   sample: dict[str, Any]) -> str:
    latency = row.get("latency_seconds")
    ram = sample.get("system_ram_available_bytes")
    ollama_ram = sample.get("ollama_process_rss_bytes")
    pieces = [f"[{index}/{total}] {chunk['paragraph_id']} chars={chunk['character_count']}",
              f"status={row['status']}", f"latency={latency:.2f}s" if latency is not None else "latency=n/a",
              f"in/out={row.get('input_tokens')}/{row.get('output_tokens')} tokens",
              f"out_tps={row.get('output_tokens_per_second'):.1f}" if row.get('output_tokens_per_second') is not None else "out_tps=n/a",
              f"RAM_available={ram / (1024 ** 2):.0f}MiB" if ram is not None else "RAM_available=n/a",
              f"Ollama_RSS={ollama_ram / (1024 ** 2):.0f}MiB" if ollama_ram is not None else "Ollama_RSS=n/a",
              f"CPU={sample.get('ollama_cpu_percent'):.0f}%" if sample.get('ollama_cpu_percent') is not None else "CPU=n/a",
              f"GPU={sample.get('gpu_utilization_percent'):.0f}%" if sample.get('gpu_utilization_percent') is not None else "GPU=n/a",
              f"swap={sample.get('swap_used_bytes') / (1024 ** 2):.0f}MiB" if sample.get('swap_used_bytes') is not None else "swap=n/a"]
    return " | ".join(pieces)


def _live_resource_line(sample: dict[str, Any]) -> str:
    available = sample.get("system_ram_available_bytes")
    ollama_rss = sample.get("ollama_process_rss_bytes")
    fields = ["[resources]"]
    fields.append(f"RAM available={available / (1024 ** 2):.0f}MiB" if available is not None else "RAM available=n/a")
    fields.append(f"Ollama RSS={ollama_rss / (1024 ** 2):.0f}MiB" if ollama_rss is not None else "Ollama RSS=n/a")
    fields.append(f"CPU={sample['ollama_cpu_percent']:.0f}%" if sample.get("ollama_cpu_percent") is not None else "CPU=n/a")
    fields.append(f"GPU={sample['gpu_utilization_percent']:.0f}%" if sample.get("gpu_utilization_percent") is not None else "GPU=n/a")
    swap = sample.get("swap_used_bytes")
    fields.append(f"swap={swap / (1024 ** 2):.0f}MiB" if swap is not None else "swap=n/a")
    return " | ".join(fields)


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["paragraph_id", "page_number", "source_page_numbers", "character_count", "input_text_sha256",
              "status", "error", "classification", "latency_seconds", "input_tokens", "output_tokens",
              "input_tokens_per_second", "output_tokens_per_second", "total_duration_ns", "load_duration_ns",
              "prompt_eval_duration_ns", "eval_duration_ns", "peak_system_ram_mb",
              "minimum_system_ram_available_mb", "peak_ollama_process_ram_mb", "mean_ollama_cpu_percent",
              "mean_system_cpu_percent", "peak_gpu_utilization_percent", "peak_gpu_memory_used_mb",
              "swap_used_mb", "swap_in_pages_delta", "swap_out_pages_delta"]
    _write_dict_rows(path, rows, fields)


def _write_dict_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else ["min_chars", "max_chars", "count", "successes", "failures"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _length_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranges = [(0, 250), (251, 500), (501, 1000), (1001, 2000), (2001, None)]
    summaries = []
    for low, high in ranges:
        group = [r for r in rows if r["character_count"] >= low and (high is None or r["character_count"] <= high)]
        if not group:
            continue
        successful = [r for r in group if r["status"] == "success"]
        summaries.append({
            "min_chars": low, "max_chars": high, "count": len(group),
            "attempted": sum(r["status"] in {"success", "failed"} for r in group),
            "successes": len(successful), "failures": sum(r["status"] == "failed" for r in group),
            "not_run": sum(r["status"].startswith("not_run") for r in group),
            "mean_latency_seconds": _mean(successful, "latency_seconds"),
            "median_latency_seconds": statistics.median([r["latency_seconds"] for r in successful if r["latency_seconds"] is not None]) if any(r["latency_seconds"] is not None for r in successful) else None,
            "mean_input_tokens": _mean(successful, "input_tokens"),
            "mean_output_tokens": _mean(successful, "output_tokens"),
            "mean_output_tokens_per_second": _mean(successful, "output_tokens_per_second"),
            "peak_system_ram_mb": _peak(group, "peak_system_ram_mb"),
            "peak_ollama_process_ram_mb": _peak(group, "peak_ollama_process_ram_mb"),
            "peak_gpu_utilization_percent": _peak(group, "peak_gpu_utilization_percent"),
            "peak_swap_used_mb": _peak(group, "swap_used_mb"),
        })
    return summaries


def _write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]],
                    groups: list[dict[str, Any]]) -> None:
    lines = ["# Local Ollama Performance Benchmark", "",
             f"- Run status: **{summary['run_status']}**", f"- Model: `{summary['model']}`",
             f"- Chunks eligible / attempted / succeeded / failed / not run: {summary['eligible_chunks']} / {summary['chunks_attempted']} / {summary['successful_chunks']} / {summary['failed_chunks']} / {summary['not_run_chunks']}",
             f"- Total runtime: {summary['total_runtime_seconds']:.2f} seconds",
             f"- Start time (UTC): {summary['start_time_utc']}",
             f"- Preflight: {'passed' if summary['preflight']['ok'] else 'failed'}",
             f"- Long chunks truncated or discarded: {summary['long_chunks_truncated_or_discarded']}", ""]
    if summary.get("stop_reason"):
        lines += [f"Stop reason: {summary['stop_reason']}", ""]
    lines += ["## Resource summary", "",
              "| Metric | Observed |", "|---|---:|",
              f"| Peak system RAM used (MiB) | {_fmt(summary.get('peak_system_ram_mb'))} |",
              f"| Minimum system RAM available (MiB) | {_fmt(summary.get('minimum_system_ram_available_mb'))} |",
              f"| Peak Ollama process RSS (MiB) | {_fmt(summary.get('peak_ollama_process_ram_mb'))} |",
              f"| Mean Ollama CPU (%) | {_fmt(summary.get('mean_ollama_cpu_percent'))} |",
              f"| Mean system CPU (%) | {_fmt(summary.get('mean_system_cpu_percent'))} |",
              f"| Peak GPU utilization (%) | {_fmt(summary.get('peak_gpu_utilization_percent'))} |",
              f"| Peak GPU memory used (MiB) | {_fmt(summary.get('peak_gpu_memory_used_mb'))} |",
              f"| Peak swap used (MiB) | {_fmt(summary.get('peak_swap_used_mb'))} |", "",
              "## Length-range summary", "",
              "| Chars | Count | Attempted | Success | Failed | Not run | Mean latency (s) | Mean output tok/s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        label = f"{group['min_chars']}–{group['max_chars']}" if group['max_chars'] is not None else f"{group['min_chars']}+"
        lines.append(f"| {label} | {group['count']} | {group['attempted']} | {group['successes']} | {group['failures']} | {group['not_run']} | {_fmt(group['mean_latency_seconds'])} | {_fmt(group['mean_output_tokens_per_second'])} |")
    lines += ["", "## Interpretation and limits", "",
              "Each eligible P04 prose unit is sent sequentially and in full to the existing OllamaDetector. Character counts are used only for ordering and grouping; inputs are not truncated. Ollama token counts and generation timing are reported only when present in the response metadata. GPU data is unavailable unless `nvidia-smi` can report it. Resource samples are periodic and may miss very brief peaks.",
              "", "This benchmark measures latency and resource use for the configured local runtime. Detector classifications are not validated authorship findings.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    return f"{float(value):.2f}" if value is not None else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark sequential local Ollama inference on eligible P04 prose.")
    parser.add_argument("--input", type=Path, default=Path("outputs/p04_category_aware_20261117T115439Z/report.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--host", default=os.getenv("AUDIT_OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--low-memory-gib", type=float, default=1.0)
    parser.add_argument("--low-memory-percent", type=float, default=5.0)
    args = parser.parse_args()
    try:
        result = run_benchmark(args.input, args.output, model=args.model, host=args.host,
                               timeout=args.timeout, minimum_available_gib=args.low_memory_gib,
                               minimum_available_percent=args.low_memory_percent)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {type(exc).__name__}: {exc}\n")
    return 0 if result["run_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
