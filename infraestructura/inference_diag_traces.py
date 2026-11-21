"""Capture and verify per-rank Kineto traces from an isolated serving job."""

from __future__ import annotations

import base64
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import tempfile
import threading
import time
import zlib

from infraestructura import inference_runtime as runtime


MAX_WINDOW_SECONDS = 60
MAX_ITERATIONS = 32
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_FILES_PER_NODE = 8
MAX_EVENTS = 250_000
MAX_RANKS = 4
CONTROL_TIMEOUT_SECONDS = 5
EXPORT_GRACE_SECONDS = 4
TRACE_DIRECTORY = "traces"

_COMPONENT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,239}")
_GLOBAL_RANK = re.compile(
    r"(?:^|[_.-])(?:global[_-])?rank[_=-]?(\d+)(?=$|[^a-zA-Z0-9])", re.I
)
_ITERATION = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
_OTHER_COMMUNICATION = re.compile(r"all_?reduce|all_?gather|all_?to_?all|reduce_?scatter|"
                                  r"collective|nvshmem|msccl|send_?recv", re.I)
_DATA_ERRORS = (OSError, ValueError, TypeError, KeyError, EOFError, OverflowError, RecursionError, zlib.error)


def server_args() -> list[str]:
    """Enable worker CPU/CUDA profiling using the torch backend."""
    return ["--profiler-config", json.dumps({
        "profiler": "torch",
        "torch_profiler_dir": "/work/traces",
        "torch_profiler_use_gzip": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_dump_cuda_time_total": False,
        "ignore_frontend": True,
        "delay_iterations": 0,
        "max_iterations": MAX_ITERATIONS,
    }, sort_keys=True, separators=(",", ":"))]


def _mark() -> dict:
    return {"monotonic_ns": time.monotonic_ns(),
            "utc": datetime.now(timezone.utc).isoformat()}


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("trace deadline exceeded")
    return remaining


def _control(port: int, action: str, deadline: float, receipt: dict) -> None:
    """Use a fixed loopback POST and an absolute deadline for response headers."""
    command = {"method": "POST", "path": f"/{action}_profile", "sent": None,
               "ack": None, "http_status": None, "error": None, "attempt": _mark()}
    receipt[action] = command
    try:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid profiler port")
        with socket.create_connection(("127.0.0.1", port), _remaining(deadline)) as conn:
            request = (f"POST {command['path']} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                       "Content-Length: 0\r\nConnection: close\r\n\r\n").encode("ascii")
            conn.settimeout(_remaining(deadline))
            command["sent"] = _mark()
            conn.sendall(request)
            headers = bytearray()
            while b"\r\n\r\n" not in headers:
                conn.settimeout(_remaining(deadline))
                chunk = conn.recv(1024)
                if not chunk or len(headers) + len(chunk) > 16 * 1024:
                    raise ValueError("incomplete or oversized profiler response headers")
                headers.extend(chunk)
            status = re.fullmatch(rb"HTTP/1\.[01] ([1-5][0-9][0-9])(?: [^\r\n]*)?",
                                  headers.split(b"\r\n", 1)[0])
            if status is None:
                raise ValueError("invalid profiler response status")
            command["http_status"] = int(status[1])
            if command["http_status"] not in (200, 204):
                raise ValueError(f"profiler HTTP {command['http_status']}")
            command["ack"] = _mark()
            if command["ack"]["monotonic_ns"] > deadline * 1e9:
                raise TimeoutError("profiler acknowledgement after control deadline")
    except (OSError, ValueError) as exc:
        command["error"] = str(exc)
    finally:
        command["finished"] = _mark()


@contextmanager
def trace_window(port: int, deadline: float):
    """Yield a mutable receipt finalized after STOP, even on client exceptions.

    The watchdog dispatches STOP with time reserved for its acknowledgement.
    An unacknowledged STOP cannot establish when the remote profiler stopped.
    A confirmed window can capture only a prefix of the client. covered_client
    describes the control envelope; the profiler's iteration cap still applies.
    Deadlines are absolute values from time.monotonic().
    """
    opened = time.monotonic()
    hard_stop = min(deadline, opened + MAX_WINDOW_SECONDS)
    reserve = min(CONTROL_TIMEOUT_SECONDS, max(0, (hard_stop - opened) / 4))
    stop_by = hard_stop - reserve
    finished = threading.Event()
    receipt = {"valid": False, "opened": _mark(), "errors": [],
               "stop_dispatch_deadline_monotonic_s": stop_by,
               "hard_deadline_monotonic_s": hard_stop}

    def stop():
        wait_seconds = max(0, stop_by - time.monotonic())
        if receipt["start"]["ack"] is None or receipt["start"]["error"] is not None:
            receipt["stop_reason"] = "start_failed"
        else:
            finished.wait(wait_seconds)
            receipt["stop_reason"] = "client_exit" if time.monotonic() < stop_by else "deadline"
        _control(port, "stop", min(hard_stop, time.monotonic() + reserve), receipt)

    # START completes or times out before STOP can be dispatched. A timeout
    # leaves the engine's state uncertain, so still attempt STOP promptly.
    _control(port, "start", min(stop_by, opened + reserve), receipt)
    watchdog = threading.Thread(target=stop, name="inference-profiler-stop", daemon=True)
    try:
        watchdog.start()
    except RuntimeError as exc:
        receipt["errors"].append(f"STOP watchdog: {exc}")
        finished.set()
        stop()
    try:
        receipt["client_started"] = _mark()
        yield receipt
    finally:
        receipt["client_finished"] = _mark()
        finished.set()
        if watchdog.ident is not None:
            watchdog.join()
        receipt["closed"] = _mark()
        receipt["errors"].extend(
            f"{action.upper()}: {receipt[action]['error']}"
            for action in ("start", "stop")
            if receipt[action]["ack"] is None or receipt[action]["error"] is not None
        )
        start, stop_result = receipt["start"], receipt["stop"]
        start_sent, stop_ack = start["sent"], stop_result["ack"]
        observed = None
        if start_sent is not None and stop_ack is not None:
            observed = (stop_ack["monotonic_ns"] - start_sent["monotonic_ns"]) / 1e9
            if not 0 <= observed <= MAX_WINDOW_SECONDS or stop_ack["monotonic_ns"] > hard_stop * 1e9:
                receipt["errors"].append("observed profiler window exceeds its deadline")
        client_start = receipt["client_started"]["monotonic_ns"]
        client_finish = receipt["client_finished"]["monotonic_ns"]
        receipt["observed_window_seconds"] = observed
        receipt["client_duration_seconds"] = (client_finish - client_start) / 1e9
        receipt["outer_duration_seconds"] = (
            receipt["closed"]["monotonic_ns"] - receipt["opened"]["monotonic_ns"]
        ) / 1e9
        receipt["valid"] = not receipt["errors"]
        receipt["covered_client"] = (
            receipt["valid"]
            and start["ack"]["monotonic_ns"] <= client_start
            and client_finish <= stop_result["sent"]["monotonic_ns"]
        )


# Invoked on the expected node through the runner's bounded transport. Transfer
# at most one new regular file per node, without tar paths or symlink traversal.
_READ_REMOTE = r"""
import base64, json, os, stat, sys
from pathlib import Path

root = Path(sys.argv[1])
previous = json.loads(sys.argv[2])
max_files, max_bytes = map(int, sys.argv[3:])
if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
    raise ValueError('invalid trace workspace')
def fingerprint(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
def collect(fd):
    entries = {}
    with os.scandir(fd) as listing:
        for index, entry in enumerate(listing):
            if index >= max_files * 4:
                raise ValueError('trace directory entry limit')
            if entry.name.endswith('.json.gz'):
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
                    raise ValueError('trace must be a regular owned workspace file')
                if len(entries) >= max_files:
                    raise ValueError('per-start trace file limit')
                entries[entry.name] = fingerprint(info)
    if any(entries.get(name) != value for name, value in previous.items()):
        raise ValueError('previous trace disappeared or changed')
    fresh = sorted(entries.keys() - previous.keys())
    if len(fresh) > 1:
        raise ValueError('multiple new traces on a single-GPU node')
    if not fresh:
        return None
    name = fresh[0]
    with os.fdopen(os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                           dir_fd=fd), 'rb') as stream:
        before = fingerprint(os.fstat(stream.fileno()))
        if before != entries[name] or not 0 < before[2] <= max_bytes:
            raise ValueError('trace changed or exceeds compressed size limit')
        content = stream.read(max_bytes + 1)
        if len(content) != before[2] or fingerprint(os.fstat(stream.fileno())) != before:
            raise ValueError('trace changed during transfer')
    return {'name': name, 'fingerprint': before, 'content': base64.b64encode(content).decode('ascii')}

root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    try:
        fd = os.open('traces', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    except FileNotFoundError:
        if previous:
            raise
        result = None
    else:
        try:
            result = collect(fd)
        finally:
            os.close(fd)
finally:
    os.close(root_fd)
print(json.dumps(result, ensure_ascii=True))
"""


def _rank_map(ranks) -> dict:
    result = {}
    ids = set()
    hosts = set()
    gpus = set()
    for record in ranks:
        rank = record["rank"]
        node = record["node_id"]
        host = record["hostname"]
        gpu = record["gpu_uuid"]
        if (type(rank) is not int or not 0 <= rank < MAX_RANKS or rank in ids
                or not isinstance(node, str) or not _COMPONENT.fullmatch(node) or node in result
                or not isinstance(host, str) or not host or host in hosts
                or not isinstance(gpu, str) or not gpu.startswith("GPU-") or gpu in gpus):
            raise ValueError("invalid or duplicate participant identity")
        ids.add(rank)
        hosts.add(host)
        gpus.add(gpu)
        result[node] = record
    if not result or ids != set(range(len(result))):
        raise ValueError("participant ranks must be contiguous from zero")
    return result


def _name_valid(name) -> bool:
    return isinstance(name, str) and bool(_COMPONENT.fullmatch(name)) and name.endswith(".json.gz")


def _fetch(node, workspace, previous, deadline):
    result = runtime.checked(runtime.remote(
        node, ["python3", "-c", _READ_REMOTE, workspace, json.dumps(previous, sort_keys=True),
               str(MAX_FILES_PER_NODE), str(MAX_COMPRESSED_BYTES)],
        timeout=min(20, _remaining(deadline)),
    ), "collect private profiler trace")
    payload = json.loads(result["stdout"])
    if payload is None:
        return None
    name, fingerprint, content = (payload[key] for key in ("name", "fingerprint", "content"))
    if (not _name_valid(name) or name in previous or not isinstance(fingerprint, list)
            or len(fingerprint) != 5 or any(type(value) is not int for value in fingerprint)
            or not 0 < fingerprint[2] <= MAX_COMPRESSED_BYTES
            or len(content) != 4 * ((fingerprint[2] + 2) // 3)):
        raise ValueError("invalid remote trace payload")
    contents = base64.b64decode(content, validate=True)
    if len(contents) != fingerprint[2]:
        raise ValueError("remote trace size mismatch")
    return name, fingerprint, contents


def collect_traces(nodes, workspace, dest: Path, ranks, seen: dict, deadline: float) -> dict:
    """Publish one verified trace per rank, as a unit, from this job's workspace.

    Initialize seen={} once per server start; keep it across cells. Any failed
    collection poisons the start because an old STOP can finish exporting late.
    Use a fresh destination under raw/diagnostic/<cell>/traces for each cell.
    """
    node_ids = []
    try:
        nodes = list(nodes)
        node_ids = [node["node_id"] for node in nodes]
        expected = _rank_map(ranks)
        if len(node_ids) != len(expected) or set(node_ids) != set(expected):
            raise ValueError("collection nodes differ from worker evidence")
        if any(node.get("expected_hostname", expected[node["node_id"]]["hostname"])
               != expected[node["node_id"]]["hostname"] for node in nodes):
            raise ValueError("collection hostnames differ from worker evidence")
        workspace = Path(workspace)
        if (not workspace.is_absolute() or ".." in workspace.parts
                or workspace == Path(workspace.anchor)):
            raise ValueError("absolute dedicated trace workspace required")
        if set(seen) - set(expected) or any(seen.get(node, {}) is None for node in node_ids):
            raise ValueError("trace attribution unavailable after incomplete capture")
        if any(len(seen.get(node, {})) >= MAX_FILES_PER_NODE for node in node_ids):
            raise ValueError("per-start trace file limit")
        if dest.exists() or dest.is_symlink():
            raise ValueError("trace destination already exists")
        _remaining(deadline)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Stage beside dest so publication is a single same-filesystem rename.
        with tempfile.TemporaryDirectory(prefix=".traces-", dir=dest.parent) as staging:
            staged = Path(staging) / TRACE_DIRECTORY
            staged.mkdir()
            files = []
            fingerprints = {}
            pending = nodes
            flush_deadline = min(deadline, time.monotonic() + EXPORT_GRACE_SECONDS)
            while pending:
                missing = []
                for node in pending:
                    node_id = node["node_id"]
                    payload = _fetch(node, str(workspace), seen.get(node_id, {}), deadline)
                    if payload is None:
                        missing.append(node)
                        continue
                    name, fingerprint, contents = payload
                    path = staged / node_id / name
                    path.parent.mkdir()
                    path.write_bytes(contents)
                    worker = expected[node_id]
                    files.append({key: worker[key] for key in ("node_id", "hostname", "gpu_uuid", "rank")}
                                 | {"path": f"{node_id}/{name}", "compressed_bytes": len(contents),
                                    "sha256": hashlib.sha256(contents).hexdigest()})
                    fingerprints[node_id] = (name, fingerprint)
                if missing:
                    if time.monotonic() >= flush_deadline:
                        names = ", ".join(node["node_id"] for node in missing)
                        raise ValueError(f"missing worker traces: {names}")
                    time.sleep(min(0.2, max(0, flush_deadline - time.monotonic())))
                pending = missing
            receipt = {"valid": True, "errors": [], "files": sorted(files, key=lambda item: item["rank"])}
            report = _verify(staged, receipt, expected, deadline)
            _remaining(deadline)
            dest.mkdir()
            try:
                staged.rename(dest)
            except OSError:
                dest.rmdir()
                raise
        for node_id, (name, fingerprint) in fingerprints.items():
            seen.setdefault(node_id, {})[name] = fingerprint
        return report
    except _DATA_ERRORS + (RuntimeError,) as exc:
        for node_id in node_ids:
            seen[node_id] = None
        return {"valid": False, "errors": [str(exc)], "files": [], "summary": []}


def validate_traces(dest: Path, receipt: dict, ranks) -> dict:
    """Recompute summaries from the archived files and independent rank evidence.

    The caller also verifies the run manifest and the separate START/STOP receipt.
    A valid collection receipt must contain complete identity and SHA-256 records.
    Activity is a union of GPU kernel intervals within each rank or stream. CPU
    ranges describe iterations and provide launch evidence, in the trace's clock.
    """
    try:
        return _verify(dest, receipt, _rank_map(ranks))
    except _DATA_ERRORS as exc:
        return {"valid": False, "errors": [str(exc)], "files": [], "summary": []}


def _verify(dest: Path, receipt: dict, expected: dict, deadline=None) -> dict:
    if not isinstance(receipt, dict) or receipt.get("valid") is not True:
        raise ValueError("collection did not attest a complete capture")
    files = receipt["files"]
    if not isinstance(files, list) or len(files) != len(expected):
        raise ValueError("trace file count differs from rank count")
    if dest.is_symlink() or not dest.is_dir() or {p.name for p in dest.iterdir()} != set(expected):
        raise ValueError("trace directory contents differ from participant set")
    summaries = []
    checked_files = []
    visited = set()
    for item in files:
        if deadline is not None:
            _remaining(deadline)
        node = item["node_id"]
        if node in visited or node not in expected:
            raise ValueError("duplicate or unknown trace node")
        worker = expected[node]
        for key in ("rank", "hostname", "gpu_uuid"):
            if type(item[key]) is not type(worker[key]) or item[key] != worker[key]:
                raise ValueError(f"trace receipt {key} differs from participant evidence")
        visited.add(node)
        relative = item["path"]
        if not isinstance(relative, str):
            raise ValueError("invalid trace path")
        name = PurePosixPath(relative).name
        parent = dest / node
        if (not _name_valid(name) or relative != f"{node}/{name}" or parent.is_symlink()
                or not parent.is_dir() or {p.name for p in parent.iterdir()} != {name}):
            raise ValueError("invalid or unlisted trace path")
        path = parent / name
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or type(item["compressed_bytes"]) is not int
                    or not 0 < info.st_size == item["compressed_bytes"] <= MAX_COMPRESSED_BYTES):
                raise ValueError("invalid compressed trace size or file type")
            contents = stream.read(MAX_COMPRESSED_BYTES + 1)
        digest = hashlib.sha256(contents).hexdigest()
        if len(contents) != info.st_size or digest != item["sha256"]:
            raise ValueError("trace SHA-256 or size mismatch")
        with gzip.GzipFile(fileobj=io.BytesIO(contents)) as compressed:
            decoded = compressed.read(MAX_UNCOMPRESSED_BYTES + 1)
        if len(decoded) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("decompressed trace size limit")
        trace = json.loads(decoded)
        del decoded
        if not isinstance(trace, dict):
            raise ValueError("Kineto trace object required")
        summaries.append(_summarize(trace, name, worker))
        del trace
        checked_files.append({key: item[key] for key in
                              ("node_id", "hostname", "gpu_uuid", "rank", "path",
                               "compressed_bytes", "sha256")})
    return {"valid": True, "errors": [], "files": sorted(checked_files, key=lambda row: row["rank"]),
            "summary": sorted(summaries, key=lambda row: row["rank"])}


def _trace_identity(trace, filename, worker):
    # The pinned image emits dpN_ppN_tpN_dcpN_epN_rankN, or rankN on fallback.
    # Never promote local_rank / tp_rank / pp_rank to a global rank. A global
    # rank can instead be supplied by Kineto's distributedInfo metadata.
    stripped = re.sub(r"(?i)(?<![a-z0-9])(?:local|tp|pp|dp|ep|pcp|dcp)[_ -]?rank", "subrank", filename)
    matched = {int(rank) for rank in _GLOBAL_RANK.findall(stripped)}
    distributed = trace.get("distributedInfo", {})
    if not isinstance(distributed, dict):
        raise ValueError("invalid distributedInfo")
    reported = distributed.get("rank")
    if reported is not None:
        if type(reported) is not int:
            raise ValueError("invalid Kineto distributed rank")
        matched.add(reported)
    if matched != {worker["rank"]}:
        raise ValueError("missing or conflicting trace global rank")
    if "world_size" in worker and "world_size" in distributed:
        if distributed["world_size"] != worker["world_size"]:
            raise ValueError("trace world size differs from worker evidence")

    hostname_source = "worker_on_collection_node"
    for field in ("hostname", "hostName"):
        if field in trace:
            if trace[field] != worker["hostname"]:
                raise ValueError("trace hostname differs from worker evidence")
            hostname_source = "trace_metadata_and_worker"

    # The runtime launches one GPU per node. Kineto generally records the
    # logical GPU index and may omit the UUID, which is then independently
    # supplied by the worker on that node.
    device = re.fullmatch(r"cuda:(\d+)", str(worker.get("device", "")))
    expected_device = None if device is None else int(device[1])
    properties = trace.get("deviceProperties", [])
    if not isinstance(properties, list) or any(not isinstance(item, dict) for item in properties):
        raise ValueError("invalid deviceProperties")
    return {
        "global_rank": "distributedInfo" if reported is not None else "filename",
        "hostname": hostname_source,
        "gpu_uuid": "single_gpu_worker",
    }, expected_device, properties


def _as_index(value):
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,10}", value):
        return int(value)
    return None


def _duration(event):
    start, duration = event.get("ts"), event.get("dur")
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (start, duration)):
        raise ValueError("invalid Kineto timestamp or duration")
    finish = start + duration
    if duration < 0 or not math.isfinite(finish):
        raise ValueError("invalid Kineto event interval")
    return start, finish


def _union(intervals):
    merged = []
    for start, finish in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], finish))
        else:
            merged.append((start, finish))
    return merged


def _activity(compute, nccl):
    compute = _union(compute)
    nccl = _union(nccl)
    left, right, overlap = 0, 0, 0
    while left < len(compute) and right < len(nccl):
        start_a, end_a = compute[left]
        start_b, end_b = nccl[right]
        overlap += max(0, min(end_a, end_b) - max(start_a, start_b))
        if end_a <= end_b:
            left += 1
        else:
            right += 1
    return {"compute_us": sum(end - start for start, end in compute),
            "nccl_us": sum(end - start for start, end in nccl), "overlap_us": overlap}


def _iteration_category(match):
    context_requests, context_tokens, generation_requests, generation_tokens = map(int, match.groups())
    context = context_requests > 0 and context_tokens > 0
    generation = generation_requests > 0 and generation_tokens > 0
    if (context_requests > 0) != (context_tokens > 0) or (generation_requests > 0) != (generation_tokens > 0):
        return "insufficient"
    if context and generation:
        return "mixed"
    if context:
        return "context"
    if generation:
        return "decode"
    return "insufficient"


def _summarize(trace, filename, worker):
    identity, expected_device, properties = _trace_identity(trace, filename, worker)
    events = trace.get("traceEvents")
    if not isinstance(events, list) or not 0 < len(events) <= MAX_EVENTS:
        raise ValueError("missing Kineto events or event count exceeds limit")
    annotations = []
    cpu_events = []
    kernels = []
    devices = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("invalid Kineto event")
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        title = event.get("name")
        if category not in ("user_annotation", "cpu_op", "cuda_runtime", "cuda_driver", "kernel"):
            continue
        if not isinstance(title, str) or not title:
            raise ValueError("missing Kineto event name")
        args = event.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("invalid Kineto event arguments")
        if category in ("user_annotation", "cpu_op"):
            match = _ITERATION.fullmatch(title)
            if match:
                start, end = _duration(event)
                annotations.append({
                    "category": _iteration_category(match), "name": title, "start_us": start,
                    "end_us": end, "pid": _as_index(event.get("pid")), "tid": _as_index(event.get("tid")),
                })
        if category == "kernel":
            device = _as_index(args.get("device"))
            if device is None:
                raise ValueError("GPU kernel without a logical device index")
            devices.add(device)
            start, end = _duration(event)
            if end > start:
                kind = "compute"
                if "nccl" in title.lower():
                    kind = "nccl"
                elif _OTHER_COMMUNICATION.search(title):
                    kind = "unclassified"
                kernels.append((start, end, _as_index(args.get("stream")), kind, args))
        elif category in ("cpu_op", "cuda_runtime", "cuda_driver"):
            cpu_events.append((event, args))

    if not kernels or len(devices) != 1:
        raise ValueError("positive GPU kernel activity on one logical device required")
    device = next(iter(devices))
    if expected_device is not None and device != expected_device:
        raise ValueError("trace device index differs from worker evidence")
    for item in properties:
        if _as_index(item.get("id")) != device:
            continue
        for key in ("uuid", "UUID"):
            if key in item:
                observed = str(item[key]).lower().removeprefix("gpu-")
                if observed != worker["gpu_uuid"].lower().removeprefix("gpu-"):
                    raise ValueError("trace GPU UUID differs from worker evidence")
                identity["gpu_uuid"] = "trace_metadata_and_worker"

    if len(annotations) > MAX_ITERATIONS:
        raise ValueError("iteration annotation count exceeds configured cap")
    annotations.sort(key=lambda row: (row["start_us"], row["end_us"], row["name"]))
    correlations, external = _launch_links(annotations, cpu_events)
    rank_intervals = {"compute": [], "nccl": []}
    streams = defaultdict(lambda: {"compute": [], "nccl": []})
    groups = defaultdict(lambda: {"compute": [], "nccl": []})
    group_counts = defaultdict(int)
    unresolved_streams = 0
    unclassified = []
    for start, finish, stream, category, args in kernels:
        interval = (start, finish)
        if stream is None:
            unresolved_streams += 1
        if category == "unclassified":
            unclassified.append(interval)
        else:
            rank_intervals[category].append(interval)
            if stream is not None:
                streams[stream][category].append(interval)
        links = (correlations.get(_as_index(args.get("correlation")), set())
                 | external.get(_as_index(args.get("External id")), set()))
        phase = annotations[next(iter(links))]["category"] if len(links) == 1 and -1 not in links else "insufficient"
        if category != "unclassified":
            groups[phase][category].append(interval)
        group_counts[phase] += 1

    return {
        "rank": worker["rank"], "node_id": worker["node_id"], "hostname": worker["hostname"],
        "gpu_uuid": worker["gpu_uuid"], "logical_device": device, "identity_evidence": identity,
        "kernel_count": len(kernels), "trace_time_unit": "us",
        "activity": _activity(**rank_intervals),
        "nccl_label_status": "observed" if rank_intervals["nccl"] else "insufficient",
        "compute_classification": "CUDA kernel events excluding recognized communication names",
        "unclassified_kernel_count": len(unclassified),
        "unclassified_kernel_us": sum(end - start for start, end in _union(unclassified)),
        "unresolved_stream_kernel_count": unresolved_streams,
        "stream_activity": [{"stream": stream, **_activity(**streams[stream])}
                            for stream in sorted(streams)],
        "iteration_status": ("correlated" if any(phase != "insufficient" for phase in group_counts)
                             else "insufficient"),
        "cpu_iteration_ranges": annotations,
        "iteration_activity": [{"category": phase, "kernel_count": group_counts[phase],
                                **_activity(**groups[phase])} for phase in sorted(groups)],
    }


def _launch_links(annotations, cpu_events):
    """Associate GPU work to CPU launches on the annotated host thread."""
    threads = defaultdict(list)
    for index, row in enumerate(annotations):
        if all(_as_index(row[key]) is not None for key in ("pid", "tid")):
            thread = (_as_index(row["pid"]), _as_index(row["tid"]))
            threads[thread].append((row["start_us"], row["end_us"], index))
    starts = {thread: [start for start, _, _ in ranges] for thread, ranges in threads.items()}
    correlations = defaultdict(set)
    external = defaultdict(set)
    for event, args in cpu_events:
        thread = (_as_index(event.get("pid")), _as_index(event.get("tid")))
        ranges = threads.get(thread)
        candidates = set()
        if ranges:
            start, end = _duration(event)
            eligible = bisect_right(starts[thread], start)
            # Overlapping host ranges stay ambiguous. Time of GPU execution alone
            # cannot establish a launch phase because CUDA work is asynchronous.
            candidates = {index for _, finish, index in ranges[:eligible] if start < finish and end <= finish}
        corr_id = _as_index(args.get("correlation"))
        ext_id = _as_index(args.get("External id"))
        if event["cat"] in ("cuda_runtime", "cuda_driver") and corr_id is not None:
            correlations[corr_id].update(candidates or {-1})
        if candidates and ext_id is not None and ext_id > 0:
            external[ext_id].update(candidates)
    return correlations, external
