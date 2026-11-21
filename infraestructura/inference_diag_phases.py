"""Evidence from vLLM 0.24 per-request Prometheus histograms."""

import base64
from datetime import datetime, timezone
import http.client
import io
import math
import os
from pathlib import Path
import re
import socket
import time


_PHASE_NAMES = {
    phase: f"vllm:request_{phase}_time_seconds" for phase in ("queue", "prefill", "decode")
}
_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
_PROCESS = "process_start_time_seconds"
_FAMILIES = (*_GAUGES, *_PHASE_NAMES.values())
# request_latency_buckets in the pinned image's vllm/v1/metrics/loggers.py.
_BUCKET_LIMITS = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0,
    30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
    math.inf,
)
_SAMPLES = {
    *_GAUGES, _PROCESS,
    *(f"{family}_{suffix}" for family in _PHASE_NAMES.values()
      for suffix in ("count", "sum", "bucket", "created")),
}
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*"((?:[^"\\]|\\[n"\\])*)"\s*')
_SAMPLE = re.compile(r'(?:\{(.*)\})?\s+(\S+)\s*')
_MAX_BODY_BYTES = 4 * 1024 * 1024
_PROC = Path("/proc")


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("metrics scrape deadline exceeded")
    return remaining


class _DeadlineReader(io.RawIOBase):
    """Set the remaining absolute deadline on every underlying HTTP read."""

    def __init__(self, sock, deadline):
        self.sock = sock
        self.deadline = deadline

    def makefile(self, mode):
        return io.BufferedReader(self)

    def readable(self):
        return True

    def readinto(self, buffer):
        self.sock.settimeout(_remaining(self.deadline))
        return self.sock.recv_into(buffer)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _listen_socket(port, deadline):
    # IPv4 endpoints from this protocol bind to 127.0.0.1 on the host netns.
    # Include wildcard binds when checking for ambiguity or a different owner.
    local_addresses = {f"{address}:{port:04X}" for address in ("0100007F", "00000000")}
    with (_PROC / "net/tcp").open(encoding="ascii") as table:
        next(table, None)
        sockets = []
        for line in table:
            _remaining(deadline)
            fields = line.split()
            if len(fields) >= 10 and fields[1] in local_addresses and fields[3] == "0A":
                sockets.append((fields[1], int(fields[7]), int(fields[9])))
    if len(sockets) != 1 or not sockets[0][0].startswith("0100007F:") or sockets[0][2] <= 0:
        raise ValueError("no unique IPv4 loopback LISTEN inode for metrics port")
    return sockets[0]


def _start_ticks(process):
    # The parenthesized command can itself contain spaces and parentheses.
    tail = (process / "stat").read_text(encoding="utf-8", errors="replace").rpartition(") ")[2]
    fields = tail.split()
    if len(fields) <= 19 or int(fields[19]) <= 0:
        raise ValueError("invalid process stat start time")
    return int(fields[19])


def _owner_fd(process, target, deadline):
    try:
        with os.scandir(process / "fd") as entries:
            for entry in entries:
                _remaining(deadline)
                try:
                    if os.readlink(entry.path) == target:
                        return Path(entry.path)
                except FileNotFoundError:
                    continue
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def _listener_identity(port, deadline):
    _remaining(deadline)
    endpoint = _listen_socket(port, deadline)
    _, uid, inode = endpoint
    target = f"socket:[{inode}]"
    boot_id = (_PROC / "sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    namespace = os.readlink(_PROC / "self/ns/net")
    owners = []
    with os.scandir(_PROC) as processes:
        for entry in processes:
            _remaining(deadline)
            if not entry.name.isascii() or not entry.name.isdecimal():
                continue
            process = Path(entry.path)
            try:
                if entry.stat(follow_symlinks=False).st_uid != uid:
                    continue
                fd = _owner_fd(process, target, deadline)
                if fd is None or os.readlink(process / "ns/net") != namespace:
                    continue
                ticks = _start_ticks(process)
                if os.readlink(fd) != target or _start_ticks(process) != ticks:
                    raise ValueError("LISTEN inode owner changed during inspection")
                owners.append((int(entry.name), ticks))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    if len(owners) != 1 or not boot_id or _listen_socket(port, deadline) != endpoint:
        raise ValueError("LISTEN inode owner unavailable, ambiguous, or changed")
    _remaining(deadline)
    pid, ticks = owners[0]
    return {"source": "listener-procfs", "port": port, "boot_id": boot_id,
            "pid": pid, "start_ticks": ticks, "socket_inode": inode}


def _bracketed_scrape(port, deadline):
    # Leave time for the HTTP scrape even if procfs enumeration is slow.
    before = after = None
    errors = []
    try:
        before = _listener_identity(port, min(deadline, time.monotonic() + 0.5))
    except (OSError, ValueError) as exc:
        errors.append(f"before GET: {exc}")
    checked_before = time.monotonic()
    snapshot = _scrape(port, deadline)
    checked_after = time.monotonic()
    if snapshot["valid"]:
        try:
            after = _listener_identity(port, min(deadline, time.monotonic() + 0.5))
        except (OSError, ValueError) as exc:
            errors.append(f"after GET: {exc}")
    if before is not None and after is not None:
        if before == after:
            snapshot["process_identity"] = {
                **before, "checked_before_monotonic_s": checked_before,
                "checked_after_monotonic_s": checked_after,
            }
        else:
            snapshot.update(valid=False, reason="LISTEN inode owner changed during GET")
    if errors:
        snapshot["process_identity_error"] = "; ".join(errors)
    return snapshot


def _scrape(port, deadline):
    snapshot = {
        "port": port,
        "url": f"http://127.0.0.1:{port}/metrics",
        "started_utc": _utc_now(),
        "started_monotonic_s": time.monotonic(),
        "http_status": None,
        "http_headers": [],
        "body": "",
        "valid": False,
        "drained": None,
    }
    body = bytearray()
    try:
        # No proxy, DNS hostname, or redirect. HTTPResponse handles chunk framing.
        with socket.create_connection(("127.0.0.1", port), _remaining(deadline)) as sock:
            request = (
                f"GET /metrics HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                "Accept: text/plain; version=0.0.4\r\n"
                "Accept-Encoding: identity\r\nConnection: close\r\n\r\n"
            )
            sock.settimeout(_remaining(deadline))
            sock.sendall(request.encode("ascii"))
            with http.client.HTTPResponse(_DeadlineReader(sock, deadline)) as response:
                response.begin()
                snapshot["http_status"] = response.status
                snapshot["http_headers"] = [list(header) for header in response.getheaders()]
                length = response.length
                while True:
                    _remaining(deadline)
                    chunk = response.read1(min(65536, _MAX_BODY_BYTES + 1 - len(body)))
                    if not chunk:
                        break
                    body.extend(chunk)
                    if len(body) > _MAX_BODY_BYTES:
                        del body[_MAX_BODY_BYTES:]
                        raise ValueError("metrics body exceeds capture limit")
                _remaining(deadline)
                if length is not None and len(body) != length:
                    raise ValueError("incomplete HTTP response body")
                if response.status != 200:
                    raise ValueError(f"metrics HTTP status {response.status}")
                if response.getheader("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("unsupported metrics content encoding")
        snapshot["valid"] = True
    except (OSError, http.client.HTTPException, ValueError) as exc:
        snapshot["reason"] = str(exc)
    snapshot["finished_monotonic_s"] = time.monotonic()
    snapshot["finished_utc"] = _utc_now()
    try:
        snapshot["body"] = body.decode("utf-8")
    except UnicodeDecodeError:
        snapshot["body"] = None
        snapshot["body_base64"] = base64.b64encode(body).decode("ascii")
        snapshot["valid"] = False
        snapshot.setdefault("reason", "metrics body is not valid UTF-8")
    return snapshot


def _labels(text):
    labels = {}
    while text:
        match = _LABEL.match(text)
        if not match or match[1] in labels:
            raise ValueError("malformed or duplicate metrics label")
        labels[match[1]] = re.sub(r'\\([n"\\])', lambda m: "\n" if m[1] == "n" else m[1], match[2])
        text = text[match.end():]
        if text and not text.startswith(","):
            raise ValueError("malformed metrics label separator")
        text = text[1:].lstrip()
    return tuple(sorted(labels.items()))


def _parse(body):
    if not isinstance(body, str):
        raise ValueError("missing UTF-8 metrics body")
    samples = {name: {} for name in _SAMPLES}
    types = {}
    units = {}
    for line in body.splitlines():
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(maxsplit=4)
            if len(parts) >= 3 and parts[1] in ("TYPE", "UNIT") and parts[2] in _FAMILIES:
                metadata = types if parts[1] == "TYPE" else units
                if len(parts) != 4 or parts[2] in metadata:
                    raise ValueError(f"malformed or duplicate metadata for {parts[2]}")
                metadata[parts[2]] = parts[3]
            continue
        name = re.split(r"[{\s]", line, maxsplit=1)[0]
        if name not in samples:
            continue
        sample = _SAMPLE.fullmatch(line[len(name):])
        if not sample:
            raise ValueError(f"malformed sample for {name}")
        labels = _labels(sample[1] or "")
        if labels in samples[name]:
            raise ValueError(f"duplicate series for {name}")
        value = float(sample[2])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid value for {name}")
        samples[name][labels] = value
    return samples, types, units


def _gauges(samples, types):
    running, waiting = (samples[name] for name in _GAUGES)
    if not running or running.keys() != waiting.keys():
        raise ValueError("running/waiting gauge series missing or unequal")
    if any(types.get(name) != "gauge" for name in _GAUGES):
        raise ValueError("running/waiting metrics must be gauges")
    for labels in running:
        tags = dict(labels)
        if tags.keys() != {"model_name", "engine"} or not tags["model_name"] or not tags["engine"].isdigit():
            raise ValueError("running/waiting gauges require model_name and numeric engine labels")
    if not all(_integer_count(value) for gauge in (running, waiting) for value in gauge.values()):
        raise ValueError("running/waiting gauges must contain integer counts")
    return running, waiting


def capture(port, deadline, model_key):
    """Poll loopback for a drained scrape before an absolute time.monotonic deadline.

    The last response is retained, including its original UTF-8 text (base64 for
    undecodable bytes). The receipt uses procfs in the local host namespace.
    Validity here covers HTTP and gauges. See phase_deltas for continuity.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid loopback port")
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("invalid model key")
    while True:
        snapshot = _bracketed_scrape(port, deadline)
        snapshot["model_key"] = model_key
        if not snapshot["valid"]:
            return snapshot
        try:
            running, waiting = _gauges(*_parse(snapshot["body"])[:2])
            if not _model_series(running, model_key):
                raise ValueError("expected model missing from running/waiting gauges")
        except ValueError as exc:
            snapshot.update(valid=False, reason=str(exc))
            return snapshot
        snapshot["running"] = sum(running.values())
        snapshot["waiting"] = sum(waiting.values())
        snapshot["drained"] = snapshot["running"] == snapshot["waiting"] == 0
        if snapshot["drained"]:
            return snapshot
        # Keep the last busy HTTP response if there is no time for another poll.
        remaining = deadline - time.monotonic()
        if remaining <= 0.1:
            snapshot.update(valid=False, reason="engine did not drain before scrape deadline")
            return snapshot
        time.sleep(0.1)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _receipt(snapshot):
    # The collector must bracket this scrape with two matching observations
    # of the actual LISTEN socket owner, from the same PID namespace/host.
    receipt = snapshot.get("process_identity")
    if not isinstance(receipt, dict) or receipt.get("source") != "listener-procfs":
        detail = snapshot.get("process_identity_error", "receipt missing")
        raise ValueError(f"missing process_start_time_seconds and listener process receipt: {detail}")
    if not isinstance(receipt.get("boot_id"), str) or not receipt["boot_id"].strip():
        raise ValueError("listener process receipt lacks boot identity")
    for field in ("pid", "start_ticks", "socket_inode", "port"):
        if type(receipt.get(field)) is not int or receipt[field] <= 0:
            raise ValueError(f"invalid listener process receipt field {field}")
    start = receipt.get("checked_before_monotonic_s")
    end = receipt.get("checked_after_monotonic_s")
    if (receipt["port"] != snapshot["port"] or not _number(start) or not _number(end)
            or not start <= snapshot["started_monotonic_s"] <= snapshot["finished_monotonic_s"] <= end):
        raise ValueError("listener process receipt does not bracket the scrape")
    return tuple(receipt[field] for field in ("boot_id", "pid", "start_ticks", "socket_inode", "port"))


def _process_source(before, after, left, right):
    old, new = left[_PROCESS], right[_PROCESS]
    if old or new:
        if len(old) != 1 or old != new or next(iter(old.values())) <= 0:
            raise ValueError("process_start_time_seconds missing, ambiguous, or changed")
    if not old or ("process_identity" in before and "process_identity" in after):
        if _receipt(before) != _receipt(after):
            raise ValueError("loopback listener process identity changed")
        return "process_start_time_seconds+listener-procfs" if old else "listener-procfs"
    return "process_start_time_seconds"


def _model_series(series, model_key):
    result = {}
    for labels, value in series.items():
        model = dict(labels).get("model_name")
        if not model:
            raise ValueError("missing model_name on vLLM metrics series")
        if model == model_key:
            result[labels] = value
    return result


def _integer_count(value):
    return value.is_integer() and value < 2**53


def _histogram(parsed, family, model_key, engines):
    samples, types, units = parsed
    if types.get(family) != "histogram" or units.get(family, "seconds") != "seconds":
        raise ValueError(f"{family} must be a histogram in seconds")
    counts, sums, buckets, created = (
        _model_series(samples[f"{family}_{suffix}"], model_key)
        for suffix in ("count", "sum", "bucket", "created")
    )
    if counts.keys() != engines or sums.keys() != engines:
        raise ValueError(f"{family} count/sum labels differ from engine gauges")
    bucket_series = {labels: {} for labels in counts}
    for labels, value in buckets.items():
        bound = dict(labels).get("le")
        group = tuple((key, val) for key, val in labels if key != "le")
        if group not in bucket_series or bound is None:
            raise ValueError(f"{family} invalid bucket labels")
        limit = math.inf if bound == "+Inf" else float(bound)
        group_buckets = bucket_series[group]
        if (bound != "+Inf" and (not math.isfinite(limit) or limit < 0)
                or any(float(existing) == limit for existing in group_buckets)):
            raise ValueError(f"{family} invalid or duplicate bucket boundary")
        group_buckets[bound] = value
    for labels, series in bucket_series.items():
        bounds = sorted(series, key=float)
        if tuple(map(float, bounds)) != _BUCKET_LIMITS:
            raise ValueError(f"{family} bucket boundaries differ from vLLM 0.24")
        ordered = [series[bound] for bound in bounds]
        if (series.get("+Inf") != counts[labels]
                or not all(_integer_count(value) for value in ordered)
                or ordered != sorted(ordered) or (counts[labels] == 0 and sums[labels] != 0)):
            raise ValueError(f"{family} incomplete or inconsistent cumulative histogram")
    if created and (created.keys() != engines or any(value <= 0 for value in created.values())):
        raise ValueError(f"{family} invalid histogram creation series")
    return counts, sums, bucket_series, created


def _phase_delta(family, left, right, model_key, engines, expected_count):
    old_count, old_sum, old_buckets, old_created = _histogram(left, family, model_key, engines)
    new_count, new_sum, new_buckets, new_created = _histogram(right, family, model_key, engines)
    if old_created != new_created:
        raise ValueError(f"{family} histogram creation series changed")
    series = []
    for labels in sorted(engines):
        if old_buckets[labels].keys() != new_buckets[labels].keys():
            raise ValueError(f"{family} bucket boundaries changed")
        count = int(new_count[labels] - old_count[labels])
        elapsed = new_sum[labels] - old_sum[labels]
        bounds = sorted(old_buckets[labels], key=float)
        bucket_delta = {
            bound: int(new_buckets[labels][bound] - old_buckets[labels][bound])
            for bound in bounds
        }
        cumulative = list(bucket_delta.values())
        if (elapsed < 0 or cumulative[0] < 0 or cumulative != sorted(cumulative)
                or (count == 0 and elapsed != 0)):
            raise ValueError(f"{family} histogram reset or inconsistent deltas")
        previous_bound = previous_count = lower = upper = 0
        for bound, subtotal in bucket_delta.items():
            observations = subtotal - previous_count
            if observations:
                lower += observations * previous_bound
                upper += observations * float(bound)
            previous_bound, previous_count = float(bound), subtotal
        # Subtracting cumulative sums can lose a few ULPs of either snapshot.
        tolerance = 4 * max(math.ulp(old_sum[labels]), math.ulp(new_sum[labels]))
        if elapsed < lower - tolerance or elapsed > upper + tolerance:
            raise ValueError(f"{family} histogram sum delta disagrees with bucket deltas")
        series.append({
            "labels": dict(labels), "count_delta": count, "sum_delta_s": elapsed,
            "bucket_deltas": bucket_delta,
        })
    count = sum(row["count_delta"] for row in series)
    infinity_count = sum(row["bucket_deltas"]["+Inf"] for row in series)
    if count != expected_count or infinity_count != expected_count:
        raise ValueError(f"{family} count/+Inf delta {count}/{infinity_count}, expected {expected_count}")
    elapsed = math.fsum(row["sum_delta_s"] for row in series)
    return {
        "count_delta": count, "inf_bucket_delta": infinity_count,
        "sum_delta_s": elapsed, "mean_s": elapsed / count, "series": series,
    }


def _measured_window(measured, model_key, expected_count):
    if (not isinstance(measured, dict) or measured.get("model_id") != model_key
            or measured.get("num_prompts") != expected_count
            or measured.get("completed") != expected_count or measured.get("failed") != 0):
        raise ValueError("missing complete measured result for the expected model and request count")
    starts, ttfts, itls = (measured.get(key) for key in ("start_times", "ttfts", "itls"))
    if any(not isinstance(rows, list) or len(rows) != expected_count for rows in (starts, ttfts, itls)):
        raise ValueError("measured result lacks per-request start, TTFT, or ITL rows")
    ends = []
    for start, ttft, intervals in zip(starts, ttfts, itls):
        if (not _number(start) or start <= 0 or not _number(ttft) or ttft <= 0
                or not isinstance(intervals, list)
                or any(not _number(value) or value < 0 for value in intervals)):
            raise ValueError("invalid measured request timestamps")
        end = math.fsum((start, ttft, math.fsum(intervals)))
        if end <= start:
            raise ValueError("invalid measured request end time")
        ends.append(end)
    start, end = min(starts), max(ends)
    duration = measured.get("duration")
    # start_times are absolute perf_counter values. Allow roundoff when
    # subtracting them to compare against vLLM's reported duration.
    if not _number(duration) or duration <= 0 or end - start > duration + 2 * math.ulp(end):
        raise ValueError("measured request span exceeds T32")
    return start, end


def phase_deltas(before, after, model_key, expected_count=32, *, measured=None):
    """Verify independent phase evidence without changing benchmark validity.

    See _receipt for the optional paired listener-procfs identity contract.
    Histogram differences are reported only when all three phases validate.
    """
    try:
        if not isinstance(model_key, str) or not model_key or type(expected_count) is not int or expected_count <= 0:
            raise ValueError("invalid model key or expected request count")
        for name, snapshot in (("before", before), ("after", after)):
            if (not isinstance(snapshot, dict) or snapshot.get("valid") is not True
                    or snapshot.get("http_status") != 200):
                reason = snapshot.get("reason") if isinstance(snapshot, dict) else None
                raise ValueError(f"{name} scrape is unavailable: {reason or 'invalid snapshot'}")
            start = snapshot.get("started_monotonic_s")
            end = snapshot.get("finished_monotonic_s")
            if not _number(start) or not _number(end) or not 0 <= start <= end:
                raise ValueError(f"{name} scrape has invalid monotonic times")
            if snapshot.get("model_key") != model_key:
                raise ValueError(f"{name} scrape model differs from the measured model")
        port = before.get("port")
        if (type(port) is not int or not 1 <= port <= 65535 or port != after.get("port")
                or before["finished_monotonic_s"] > after["started_monotonic_s"]):
            raise ValueError("metrics snapshots differ in port or overlap in time")
        batch_start, batch_end = _measured_window(measured, model_key, expected_count)
        if not before["finished_monotonic_s"] <= batch_start < batch_end <= after["started_monotonic_s"]:
            raise ValueError("measured request timestamps fall outside the metrics scrape window")
        left, right = (_parse(snapshot.get("body")) for snapshot in (before, after))
        process_source = _process_source(before, after, left[0], right[0])
        engines = []
        gauge_labels = []
        for parsed in (left, right):
            running, waiting = _gauges(*parsed[:2])
            if any(running.values()) or any(waiting.values()):
                raise ValueError("engine was not drained at both snapshots")
            engines.append(_model_series(running, model_key).keys())
            gauge_labels.append(running.keys())
        if not engines[0] or engines[0] != engines[1] or gauge_labels[0] != gauge_labels[1]:
            raise ValueError("model gauge series missing or changed")
        phases = {
            phase: _phase_delta(family, left, right, model_key, engines[0], expected_count)
            for phase, family in _PHASE_NAMES.items()
        }
    except (ValueError, OverflowError) as exc:
        return {"valid": False, "reason": str(exc), "phases": {}}
    return {
        "valid": True, "model_key": model_key, "expected_count": expected_count,
        "measured_request_window_monotonic_s": [batch_start, batch_end],
        "process_identity_source": process_source, "phases": phases,
    }
