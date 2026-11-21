"""Cell-local execution checks against sealed diagnostic run artifacts."""

from datetime import datetime
import json
import os
from pathlib import Path
import re

from infraestructura import manifest


_MAX_LOG_WINDOW_BYTES = 8 * 1024 * 1024
_MIN_MEMORY_KIB = 8 * 1024 * 1024
_EDGE_GAP_SECONDS = 3
_MAX_GAP_SECONDS = 10
_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*")
_COMPILATION = re.compile(rb"compil(?:ing|ation)|captur(?:ing|e).*cuda.*graph", re.I)
_PROFILER_STOP_FAILURE = re.compile(rb"\bfailed to stop profiler\b", re.I)
_EVIDENCE_ERRORS = (OSError, ValueError, TypeError, KeyError, OverflowError,
                    RecursionError, manifest.ManifestError)


def profiler_stop_failed(text: str | bytes) -> bool:
    """Identify the explicit profiler stop failure in one log or byte window."""
    if isinstance(text, str):
        text = text.encode("utf-8", errors="replace")
    return _PROFILER_STOP_FAILURE.search(text) is not None


def _utc(value):
    time = datetime.fromisoformat(value)
    if time.utcoffset() is None:
        raise ValueError("timestamp without UTC offset")
    return time


def _participants(ranks):
    ranks = list(ranks)
    by_node = {}
    for worker in ranks:
        node_id, host, gpu, rank, world = (
            worker[key] for key in ("node_id", "hostname", "gpu_uuid", "rank", "world_size")
        )
        if (not isinstance(node_id, str) or not _IDENTIFIER.fullmatch(node_id)
                or node_id in by_node or not isinstance(host, str) or not host
                or not isinstance(gpu, str) or not gpu.startswith("GPU-")
                or type(rank) is not int or type(world) is not int or world != len(ranks)):
            raise ValueError("invalid worker identity or incomplete world")
        by_node[node_id] = worker
    if ("pegasus" not in by_node
            or {w["rank"] for w in ranks} != set(range(len(ranks)))
            or len({w["hostname"] for w in ranks}) != len(ranks)
            or len({w["gpu_uuid"] for w in ranks}) != len(ranks)):
        raise ValueError("server or observed ranks missing or duplicated")
    return by_node


def _slice(path, bounds):
    if not isinstance(bounds, list) or len(bounds) != 2 or any(type(i) is not int for i in bounds):
        raise ValueError("byte offsets must be [start, end]")
    start, end = bounds
    if not 0 <= start <= end or end - start > _MAX_LOG_WINDOW_BYTES:
        raise ValueError("unordered, negative or oversized log offsets")
    with path.open("rb") as stream:
        if end > os.fstat(stream.fileno()).st_size:
            raise ValueError("log offsets exceed archived size")
        for boundary in {start, end} - {0}:
            stream.seek(boundary - 1)
            if stream.read(1) != b"\n":
                raise ValueError("log offset cuts a line")
        stream.seek(start)
        text = stream.read(end - start)
        if len(text) != end - start:
            raise ValueError("archived log window is truncated")
    return text


def _log_issues(path, cell_name, start, end, by_node):
    receipt_path = f"raw/diagnostic/{cell_name}/execution-window.json"
    try:
        source = manifest._internal_file(path, receipt_path)
        receipt = json.loads(source.read_text(encoding="utf-8"))
        if (receipt["cell"] != cell_name or _utc(receipt["measurement_started_at"]) != start
                or _utc(receipt["measurement_ended_at"]) != end):
            raise ValueError("receipt belongs to another cell or measured interval")
        logs = receipt["logs"]
        if not isinstance(logs, dict) or set(logs) != set(by_node):
            raise ValueError("execution-window log nodes differ from the observed world")
    except _EVIDENCE_ERRORS as exc:
        return [f"execution-window: {exc}"]

    issues = []
    for node, worker in sorted(by_node.items()):
        expected = {f"worker-rank{worker['rank']:02d}.log"}
        if node == "pegasus":
            expected.add("server.log")
        observed = logs[node]
        if not isinstance(observed, dict) or set(observed) != expected:
            issues.append(f"{node}: execution-window log set differs from server/worker evidence")
        if not isinstance(observed, dict):
            continue
        for name in sorted(expected.intersection(observed)):
            relative = f"raw/nodes/{node}/logs/{name}"
            try:
                log = manifest._internal_file(path, relative)
                text = _slice(log, observed[name])
                if _COMPILATION.search(text):
                    issues.append(f"{node}/{name}: compilation or CUDA graph capture in measured log window")
                if profiler_stop_failed(text):
                    issues.append(f"{node}/{name}: Failed to stop profiler in measured log window")
            except _EVIDENCE_ERRORS as exc:
                issues.append(f"{node}/{name}: {exc}")
    return issues


def _check_sample(sample, worker):
    if sample["hostname"] != worker["hostname"]:
        raise ValueError("hostname differs from observed participant")
    gpus = sample["gpus"]
    if (not isinstance(gpus, list)
            or sum(gpu["uuid"] == worker["gpu_uuid"] for gpu in gpus) != 1):
        raise ValueError("expected GPU UUID absent or duplicated")
    # Unsupported optional NVIDIA metrics are stored as null.
    memory = sample["memory"]["MemAvailable_kib"]
    if type(memory) is not int or memory < 0:
        raise ValueError("invalid MemAvailable_kib")
    if memory < _MIN_MEMORY_KIB:
        raise ValueError("memory pressure below 8 GiB MemAvailable")


def telemetry_window(path, start: datetime, end: datetime, worker) -> list[dict]:
    """Read and validate samples inside the inclusive measured UTC interval.

    Each returned sample includes _utc_seconds for the telemetry summarizer.
    An unreadable row has unknown time until bracketed by readable clocks.
    Raise ValueError if its possible interval overlaps the measured window.
    """
    if start.utcoffset() is None or end.utcoffset() is None or end <= start:
        raise ValueError("telemetry requires an increasing, timezone-aware measured interval")
    window = []
    previous_clock = None
    damaged_line = None
    # Decode rows independently. A text stream can fail on decoding its buffer
    # before returning earlier intact lines, even when the damage is in the tail.
    with Path(path).open("rb") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                utc = _utc(sample["timestamp_utc"])
                monotonic = sample["monotonic_ns"]
                if type(monotonic) is not int or monotonic < 0:
                    raise ValueError("invalid monotonic timestamp")
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                if damaged_line is None:
                    damaged_line = line_no
                continue

            previous_utc = previous_clock[0] if previous_clock is not None else None
            if previous_clock is not None and (utc < previous_utc or monotonic <= previous_clock[1]):
                # Only disorder whose two UTC bounds touch the measured window
                # affects this cell. Keep scanning after a fault outside it.
                if min(utc, previous_utc) <= end and max(utc, previous_utc) >= start:
                    raise ValueError(f"line {line_no}: non-monotonic telemetry at measured window")
            if damaged_line is not None:
                if utc > start and (previous_utc is None or previous_utc < end):
                    raise ValueError(f"line {damaged_line}: unreadable telemetry overlaps measured window")
                damaged_line = None
            if utc > end:
                break
            previous_clock = (utc, monotonic)
            if utc < start:
                continue

            try:
                _check_sample(sample, worker)
                sample["_utc_seconds"] = utc.timestamp()
                if window:
                    utc_gap = sample["_utc_seconds"] - window[-1]["_utc_seconds"]
                    mono_gap = (monotonic - window[-1]["monotonic_ns"]) / 1e9
                    if max(utc_gap, mono_gap) > _MAX_GAP_SECONDS:
                        raise ValueError("telemetry gap exceeds 10 s in measured window")
            except _EVIDENCE_ERRORS as exc:
                raise ValueError(f"line {line_no}: {exc}") from exc
            window.append(sample)

    if damaged_line is not None and (previous_clock is None or previous_clock[0] < end):
        raise ValueError(f"line {damaged_line}: unbracketed unreadable telemetry at measured window")
    if not window:
        raise ValueError("no samples inside measured window")
    if max(window[0]["_utc_seconds"] - start.timestamp(),
           end.timestamp() - window[-1]["_utc_seconds"]) > _EDGE_GAP_SECONDS:
        raise ValueError("telemetry does not cover measured window edges within 3 s")
    return window


def cell_issues(path, cell, ranks) -> list[str]:
    """Verify logs and telemetry for one cell without writing or operating the run.

    The caller verifies manifest hashes and passes the observed load-model rank
    records. The receipt binds half-open byte ranges in the final node logs to
    this cell's measured timestamps. The server log lives on Pegasus.
    """
    try:
        path = Path(path)
        cell_name = cell["cell"]
        if not isinstance(cell_name, str) or not _IDENTIFIER.fullmatch(cell_name):
            raise ValueError("invalid cell ID")
        start, end = (_utc(cell[key]) for key in ("measurement_started_at", "measurement_ended_at"))
        if end <= start:
            raise ValueError("measured interval must increase")
        by_node = _participants(ranks)
    except _EVIDENCE_ERRORS as exc:
        return [f"execution quality: {exc}"]

    issues = _log_issues(path, cell_name, start, end, by_node)
    for node, worker in sorted(by_node.items()):
        try:
            source = manifest._internal_file(path, f"raw/nodes/{node}/logs/telemetry.jsonl")
            telemetry_window(source, start, end, worker)
        except _EVIDENCE_ERRORS as exc:
            issues.append(f"{node}/telemetry.jsonl: {exc}")
    return issues
