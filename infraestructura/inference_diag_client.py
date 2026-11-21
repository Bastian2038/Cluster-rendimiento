"""Optional timing evidence for the measured vLLM completions client."""

import hashlib
from importlib import import_module, metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from types import CodeType
import uuid


SERVE_SHA256 = "b09e85eed6f5f7a9075b40a19fc1dd8e94a1c449665c2c6510f349e0409e77b7"
ENDPOINT_SHA256 = "5fd4b66d9ac6a88a278c2b7d65dcd17b8a9424279859ade14bddb69e342a312d"
EVENTS_FILENAME = "client-events.json"
PROFILE_FILENAME = "profile-control.json"
_ORIGINAL = ("vllm", "bench", "serve")
_MEASURED_FILENAME = "measured.json"
_COUNT = 32
_CLOCK = "client-perf-counter-relative-to-benchmark-start"
_PROFILE_OPTION = "--ipre-profile-port"
_MONOTONIC_CLOCK = "clock_gettime(CLOCK_MONOTONIC)"
_PATCHES = (
    (
        "async def limited_request_func(request_func_input, session, pbar):\n"
        "        async with semaphore:\n",
        "async def limited_request_func(request_func_input, session, pbar, diag_row):\n"
        "        async with semaphore:\n"
        "            _ipre_diag_acquired(diag_row, benchmark_start_time)\n",
    ),
    (
        "        tasks.append(\n            asyncio.create_task(\n"
        "                limited_request_func(\n"
        "                    request_func_input=request_func_input, session=session, pbar=pbar\n",
        "        diag_row = _ipre_diag_created(benchmark_start_time, request_func_input)\n"
        "        tasks.append(\n            asyncio.create_task(\n"
        "                limited_request_func(\n"
        "                    request_func_input=request_func_input, session=session, pbar=pbar,\n"
        "                    diag_row=diag_row,\n",
    ),
    (
        "    benchmark_start_time = time.perf_counter()\n",
        "    _ipre_diag_before_batch()\n"
        "    benchmark_start_time = time.perf_counter()\n",
    ),
    (
        "    benchmark_duration = time.perf_counter() - benchmark_start_time\n",
        "    benchmark_duration = time.perf_counter() - benchmark_start_time\n"
        "    _ipre_diag_after_batch(benchmark_start_time, benchmark_duration)\n",
    ),
)
_ENDPOINT_PATCHES = (
    (
        "                                    first_chunk_received = True\n"
        "                                    ttft = time.perf_counter() - st\n",
        "                                    first_chunk_received = True\n"
        "                                    first_token_timestamp = timestamp\n"
        "                                    ttft = time.perf_counter() - st\n",
    ),
    (
        "                if first_chunk_received:\n"
        "                    output.success = True\n",
        "                if first_chunk_received:\n"
        "                    _ipre_diag_stream(request_func_input, st, first_token_timestamp, "
        "most_recent_timestamp)\n"
        "                    output.success = True\n",
    ),
)


def _option(argv, name):
    values = []
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{name} must have a value")
            values.append(argv[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg.partition("=")[2])
    if len(values) > 1:
        raise ValueError(f"{name} must be unambiguous")
    return values[0] if values else None


def _result_path(args):
    directory = _option(args, "--result-dir")
    if _option(args, "--result-filename") != _MEASURED_FILENAME or not directory:
        raise ValueError("auxiliary client requires --result-dir and --result-filename measured.json")
    if _option(args, "--backend") != "openai" or _option(args, "--endpoint") != "/v1/completions":
        raise ValueError("auxiliary client requires the OpenAI completions backend")
    return Path(directory) / _MEASURED_FILENAME


def _check_port(port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid profiler loopback port")
    return port


def command(argv, profile_port=None):
    """Replace only the interior measured client entry point, preserving arguments."""
    prefix = tuple(argv[:3])
    args = list(argv[3:])
    if prefix != _ORIGINAL:
        raise ValueError("expected an interior vllm bench serve command")
    _result_path(args)
    prefix = ["python3", "-m", "infraestructura.inference_diag_client"]
    if profile_port is not None:
        prefix.append(f"{_PROFILE_OPTION}={_check_port(profile_port)}")
    return [*prefix, *args]


def _replace_source(source, filename, sha256, patches):
    if hashlib.sha256(source).hexdigest() != sha256:
        raise ValueError(f"installed vLLM {Path(filename).name} SHA-256 differs from archived source")
    patched = source.decode("utf-8")
    for old, new in patches:
        if patched.count(old) != 1:
            raise ValueError("ambiguous instrumented benchmark source location")
        patched = patched.replace(old, new)
    return compile(patched, filename, "exec", dont_inherit=True)


def _instrumented_code(source, filename):
    module_code = _replace_source(source, filename, SERVE_SHA256, _PATCHES)
    return _function_code(module_code, "benchmark")


def _function_code(module_code, name):
    return next(code for code in module_code.co_consts
                if isinstance(code, CodeType) and code.co_name == name)


def _clock_identity():
    return {
        "pid": os.getpid(), "invocation_id": uuid.uuid4().hex,
        "perf_counter": time.get_clock_info("perf_counter").implementation,
        "monotonic": time.get_clock_info("monotonic").implementation,
    }


def _common_clock(identity):
    return (isinstance(identity, dict)
            and type(identity.get("pid")) is int and identity["pid"] > 0
            and isinstance(identity.get("invocation_id"), str) and bool(identity["invocation_id"])
            and identity.get("perf_counter") == identity.get("monotonic") == _MONOTONIC_CLOCK)


def _acknowledged(action):
    return (isinstance(action, dict) and isinstance(action.get("ack"), dict)
            and type(action["ack"].get("monotonic_ns")) is int
            and action.get("error") is None and action.get("http_status") in (200, 204))


class _ProfileWindow:
    def __init__(self, port, clock):
        self.port = port
        self.clock = clock
        self.context = None
        self.enter_failed = False
        self.deadline = None
        self.receipt = {"valid": False, "errors": [], "requires_server_shutdown": False}
        self.origin = self.duration = None

    def open(self):
        if not _common_clock(self.clock):
            self.receipt["errors"].append("client clocks do not share CLOCK_MONOTONIC")
            return
        try:
            from infraestructura.inference_diag_traces import trace_window
        except Exception as exc:
            self.receipt["errors"].append(f"profile control unavailable: {exc}")
            return
        try:
            self.deadline = time.monotonic() + 60
            self.context = trace_window(self.port, deadline=self.deadline)
            self.receipt = self.context.__enter__()
            self.receipt["requires_server_shutdown"] = False
        except BaseException as exc:
            self.context = None
            self.enter_failed = True
            self.receipt["errors"].append(f"could not establish profiler state: {exc}")
            self.receipt["requires_server_shutdown"] = True
            raise
        if not _acknowledged(self.receipt.get("start")):
            self.close()
            if self.receipt["requires_server_shutdown"]:
                raise RuntimeError("profiler START state uncertain without STOP; close the server")

    def finish(self, origin, duration):
        self.origin, self.duration = origin, duration
        self.close()

    def close(self, exc_info=(None, None, None)):
        if self.enter_failed:
            self.enter_failed = False
            # A context interrupted during entry may have sent START before
            # installing its watchdog. Bound this recovery by the same window.
            try:
                from infraestructura.inference_diag_traces import _control, CONTROL_TIMEOUT_SECONDS
                now = time.monotonic()
                _control(self.port, "stop", min(self.deadline, now + CONTROL_TIMEOUT_SECONDS), self.receipt)
            except Exception as exc:
                self.receipt["errors"].append(f"emergency STOP failed: {exc}")
            self.receipt["requires_server_shutdown"] = not _acknowledged(self.receipt.get("stop"))
            return
        if self.context is None:
            return
        context, self.context = self.context, None
        try:
            context.__exit__(*exc_info)
        except Exception as exc:
            self.receipt["valid"] = False
            self.receipt["errors"].append(f"could not finalize profiler control: {exc}")
        finally:
            start = self.receipt.get("start", {})
            self.receipt["requires_server_shutdown"] = (
                start.get("sent") is not None and not _acknowledged(self.receipt.get("stop"))
            )

    def evidence(self, path, reason):
        if reason:
            self.receipt["errors"].append(reason)
        if self.receipt["requires_server_shutdown"]:
            self.receipt["errors"].append("profiler STOP unconfirmed; close the server")
        if self.origin is None or reason or self.receipt["requires_server_shutdown"]:
            self.receipt["valid"] = False
        self.receipt.update(
            client_clock=self.clock, benchmark_start_perf_counter_s=self.origin,
            benchmark_duration_s=self.duration, profile_port=self.port,
        )
        if self.origin is not None:
            try:
                self.receipt["measured_sha256"] = _fingerprint(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as exc:
                self.receipt["valid"] = False
                self.receipt["errors"].append(f"measured result unavailable: {exc}")
        return self.receipt


class _Recorder:
    def __init__(self, profile_port=None):
        self.rows = []
        self.origin = None
        self.active_rows = {}
        self.clock = _clock_identity()
        self.profile = None if profile_port is None else _ProfileWindow(profile_port, self.clock)

    def before_batch(self):
        if self.profile is not None:
            self.profile.open()

    def after_batch(self, origin, duration):
        if self.profile is not None:
            self.profile.finish(origin, duration)

    def created(self, origin, request):
        # The original gather preserves tasks.append order, which is the
        # order of every per-request array in the original detailed JSON.
        if self.origin is None:
            self.origin = origin
        row = {
            "row_index": len(self.rows),
            "request_id": request.request_id,
            "task_created_offset_s": time.perf_counter() - origin,
        }
        self.rows.append(row)
        self.active_rows[id(request)] = row
        return row

    def acquired(self, row, origin):
        row["semaphore_acquired_offset_s"] = time.perf_counter() - origin

    def stream(self, request, start, first, last):
        row = self.active_rows.pop(id(request), None)
        if row is not None:
            row.update(
                request_started_offset_s=start - self.origin,
                first_token_offset_s=first - self.origin,
                last_token_offset_s=last - self.origin,
            )

    def install(self, serve):
        endpoint = import_module("vllm.benchmarks.lib.endpoint_request_func")
        source_file = Path(serve.__file__)
        serve_code = _instrumented_code(source_file.read_bytes(), str(source_file))
        endpoint_file = Path(endpoint.__file__)
        endpoint_module_code = _replace_source(
            endpoint_file.read_bytes(), str(endpoint_file), ENDPOINT_SHA256, _ENDPOINT_PATCHES
        )
        request_func = endpoint.async_request_openai_completions
        endpoint_code = _function_code(endpoint_module_code, request_func.__name__)
        if (serve.benchmark.__code__.co_freevars != serve_code.co_freevars
                or request_func.__code__.co_freevars != endpoint_code.co_freevars
                or serve.ASYNC_REQUEST_FUNCS["openai"] is not request_func):
            raise ValueError("installed vLLM benchmark or endpoint function is incompatible")
        serve.benchmark.__globals__.update(
            _ipre_diag_created=self.created, _ipre_diag_acquired=self.acquired,
            _ipre_diag_before_batch=self.before_batch, _ipre_diag_after_batch=self.after_batch,
        )
        request_func.__globals__["_ipre_diag_stream"] = self.stream
        serve.benchmark.__code__ = serve_code
        request_func.__code__ = endpoint_code

    def evidence(self, measured):
        result = {
            "schema_version": 1,
            "clock": _CLOCK,
            "client_clock": self.clock,
            "benchmark_start_perf_counter_s": self.origin,
            "serve_source_sha256": SERVE_SHA256,
            "endpoint_source_sha256": ENDPOINT_SHA256,
            "measured_sha256": _fingerprint(measured),
            "rows": self.rows,
        }
        result.update(verify_events(result, measured))
        return result


def _fingerprint(measured):
    # Stable across independent reads of measured.json, including key order.
    canonical = json.dumps(measured, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _number(value, label, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"invalid {label}")
    return value


def _verify_stream_row(measured, index, origin, start, first, last):
    # The backend records TTFT on a second perf_counter call in the first
    # SSE choices branch. The first/last choice timestamps are used for ITL.
    origin = _number(origin, "benchmark clock origin", positive=True)
    measured_start = _number(measured["start_times"][index], "request start time") - origin
    ttft = _number(measured["ttfts"][index], "TTFT", positive=True)
    itls = measured["itls"][index]
    if not isinstance(itls, list):
        raise ValueError("invalid ITL row")
    intervals = math.fsum(_number(interval, "ITL interval") for interval in itls)
    if (not math.isclose(measured_start, start, abs_tol=1e-9, rel_tol=0)
            or not -1e-9 <= start + ttft - first <= 0.001
            or not math.isclose(last - first, intervals, abs_tol=1e-9, rel_tol=0)):
        raise ValueError(f"client stream offsets disagree with measured row {index}")


def verify_events(events, measured):
    """Verify row mapping, clocks and provenance against original detailed JSON."""
    try:
        if not isinstance(events, dict) or not isinstance(measured, dict):
            raise ValueError("missing client events or detailed benchmark")
        if events.get("valid") is False:
            raise ValueError(events.get("reason") or "client events were marked invalid")
        if (events.get("schema_version") != 1 or events.get("serve_source_sha256") != SERVE_SHA256
                or events.get("endpoint_source_sha256") != ENDPOINT_SHA256
                or events.get("clock") != _CLOCK or not _common_clock(events.get("client_clock"))):
            raise ValueError("client event schema, source hash, or clock differs")
        if events.get("measured_sha256") != _fingerprint(measured):
            raise ValueError("client events do not match measured benchmark")
        if (measured.get("backend") != "openai" or measured.get("completed") != _COUNT
                or measured.get("num_prompts") != _COUNT or measured.get("failed") != 0):
            raise ValueError("expected 32 completed OpenAI requests with no failures")
        concurrency = measured.get("max_concurrency")
        if type(concurrency) is not int or concurrency <= 0:
            raise ValueError("missing positive client semaphore limit")
        duration = _number(measured.get("duration"), "benchmark duration", positive=True)
        rows = events.get("rows")
        for key, values in (("rows", rows), *((key, measured.get(key)) for key in (
                "start_times", "ttfts", "itls", "errors", "output_lens"))):
            if not isinstance(values, list) or len(values) != _COUNT:
                raise ValueError(f"expected {_COUNT} entries in {key}")
        if any(measured["errors"]):
            raise ValueError("measured requests report errors")

        request_ids = set()
        previous_creation = 0
        semaphore_edges = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or type(row.get("row_index")) is not int or row["row_index"] != index:
                raise ValueError(f"invalid client event row index {index}")
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip() or request_id in request_ids:
                raise ValueError(f"missing or duplicate vLLM request ID at row {index}")
            request_ids.add(request_id)
            ordered = [
                _number(row.get(key), key) for key in (
                    "task_created_offset_s", "semaphore_acquired_offset_s",
                    "request_started_offset_s", "first_token_offset_s", "last_token_offset_s",
                )
            ]
            created, acquired, start, first, last = ordered
            if not previous_creation <= created <= acquired <= start <= first <= last <= duration:
                raise ValueError(f"client timestamps out of order or outside T32 at row {index}")
            _verify_stream_row(measured, index, events.get("benchmark_start_perf_counter_s"),
                               start, first, last)
            output_len = measured["output_lens"][index]
            if type(output_len) is not int or output_len <= 0:
                raise ValueError(f"invalid output length at row {index}")
            if acquired < last:
                semaphore_edges.extend(((acquired, 1), (last, -1)))
            previous_creation = created
        active = 0
        # At a shared boundary -1 sorts before +1 and releases the old slot.
        for _, change in sorted(semaphore_edges):
            active += change
            if active > concurrency:
                raise ValueError(f"client event intervals exceed semaphore limit {concurrency}")
    except (ValueError, TypeError, OverflowError) as exc:
        return {"valid": False, "reason": str(exc)}
    return {"valid": True, "count": _COUNT}


def profile_coverage(control, events, measured):
    """Count request streams touched or fully enclosed by confirmed control marks."""
    try:
        validated = verify_events(events, measured)
        if not validated["valid"]:
            raise ValueError(validated["reason"])
        if not isinstance(control, dict) or control.get("valid") is not True:
            raise ValueError("profile control is missing or invalid")
        clock = control.get("client_clock")
        if not _common_clock(clock) or clock != events.get("client_clock"):
            raise ValueError("profile control and events lack a shared client CLOCK_MONOTONIC")
        origin = events["benchmark_start_perf_counter_s"]
        duration = measured["duration"]
        if (control.get("measured_sha256") != events["measured_sha256"]
                or control.get("benchmark_start_perf_counter_s") != origin
                or control.get("benchmark_duration_s") != duration):
            raise ValueError("profile control and events refer to different measured batches")
        start, stop = control.get("start"), control.get("stop")
        if (control.get("requires_server_shutdown") is not False
                or not _acknowledged(start) or not _acknowledged(stop)):
            raise ValueError("profile START and STOP must both be confirmed")
        sent_ns = start["sent"]["monotonic_ns"]
        begin_ns = start["ack"]["monotonic_ns"]
        end_ns = stop["sent"]["monotonic_ns"]
        stop_ack_ns = stop["ack"]["monotonic_ns"]
        if (any(type(mark) is not int for mark in (sent_ns, begin_ns, end_ns, stop_ack_ns))
                or not 0 < sent_ns <= begin_ns < end_ns <= stop_ack_ns
                or stop_ack_ns - sent_ns > 60_000_000_000
                or begin_ns / 1e9 > origin):
            raise ValueError("profiler marks are out of order, exceed 60 seconds, or START follows T32 onset")
        begin, end = begin_ns / 1e9 - origin, end_ns / 1e9 - origin
        if control.get("stop_reason") == "client_exit" and end_ns / 1e9 < origin + duration:
            raise ValueError("profiler STOP for client_exit was sent before the T32 end")
        count = fully = 0
        for row in events["rows"]:
            request_start = row["request_started_offset_s"]
            request_end = row["last_token_offset_s"]
            if max(begin, request_start) < min(end, request_end):
                count += 1
                fully += begin <= request_start and request_end <= end
        if not count:
            raise ValueError("confirmed profiler window overlaps no measured request stream")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"valid": False, "count": 0, "fully_covered_count": 0, "reason": str(exc)}
    return {"valid": True, "count": count, "fully_covered_count": fully}


def _entry_point():
    return next(entry for entry in metadata.distribution("vllm").entry_points
                if entry.group == "console_scripts" and entry.name == "vllm").load()


def _save_events(path, recorder, reason):
    try:
        if reason is None:
            try:
                measured = json.loads(path.read_text(encoding="utf-8"))
                events = recorder.evidence(measured)
            except (OSError, ValueError, KeyError, IndexError, TypeError, OverflowError) as exc:
                reason = f"cannot verify client evidence: {exc}"
        if reason is not None:
            events = {"schema_version": 1, "valid": False, "reason": reason}
        destination = path.with_name(EVENTS_FILENAME)
        try:
            encoded = json.dumps(events, indent=2, allow_nan=False)
        except (ValueError, TypeError) as exc:
            encoded = json.dumps({"schema_version": 1, "valid": False,
                                  "reason": f"cannot serialize client evidence: {exc}"})
        with destination.open("x", encoding="utf-8") as output:
            output.write(encoded + "\n")
    except (OSError, ValueError, TypeError) as exc:
        print(f"auxiliary client evidence could not be saved: {exc}", file=sys.stderr)


def main(args=None):
    args = sys.argv[1:] if args is None else list(args)
    profile_port = None
    if args and args[0].startswith(_PROFILE_OPTION + "="):
        profile_port = _check_port(int(args[0].partition("=")[2]))
        args = args[1:]
    path = _result_path(args)
    sys.dont_write_bytecode = True
    sys.argv[:] = [*_ORIGINAL, *args]
    recorder = _Recorder(profile_port)
    reason = None
    try:
        recorder.install(import_module("vllm.benchmarks.serve"))
    except Exception as exc:
        reason = f"client instrumentation unavailable: {exc}"

    # Invoke the real console entry point exactly once, even if the optional
    # instrumentation is unavailable. Preserve both its result and exit code.
    completed = False
    try:
        status = _entry_point()()
        completed = status is None or status == 0
        return status
    except SystemExit as exc:
        completed = exc.code is None or exc.code == 0
        raise
    finally:
        reason = reason if completed else "original vLLM client failed"
        if recorder.profile is not None:
            recorder.profile.close(sys.exc_info())
            try:
                control = recorder.profile.evidence(path, reason)
                with path.with_name(PROFILE_FILENAME).open("x", encoding="utf-8") as output:
                    json.dump(control, output, indent=2, allow_nan=False)
                    output.write("\n")
            except (OSError, ValueError, TypeError) as exc:
                print(f"profile control receipt unavailable: {exc}", file=sys.stderr)
        _save_events(path, recorder, reason)


if __name__ == "__main__":
    sys.exit(main())
