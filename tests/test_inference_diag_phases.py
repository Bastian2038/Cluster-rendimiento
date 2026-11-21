import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import contextmanager

from infraestructura import inference_diag_client as client
from infraestructura import inference_diag_phases as phases


PORT = 18000
MODEL = "qwen35-9b"


def metrics(count, *, model=MODEL, engine=0, running=0, waiting=0, process_start=None):
    labels = f'engine="{engine}",model_name="{model}"'
    lines = [
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{{{labels}}} {running}",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{{{labels}}} {waiting}",
    ]
    if process_start is not None:
        lines.append(f"process_start_time_seconds {process_start}")
    for phase in ("queue", "prefill", "decode"):
        metric = f"vllm:request_{phase}_time_seconds"
        lines.append(f"# TYPE {metric} histogram")
        for bound in phases._BUCKET_LIMITS:
            boundary = "+Inf" if bound == float("inf") else str(bound)
            subtotal = count // 2 if bound == 0.3 else count
            lines.append(f'{metric}_bucket{{le="{boundary}",{labels}}} {subtotal}')
        lines.extend((
            f"{metric}_count{{{labels}}} {count}",
            f"{metric}_sum{{{labels}}} {count * 0.25}",
        ))
    return "\n".join(lines) + "\n"


def snapshot(body, start):
    return {
        "valid": True, "body": body, "http_status": 200, "port": PORT, "model_key": MODEL,
        "started_monotonic_s": start, "finished_monotonic_s": start + 0.5,
    }


def phase_measurement(before, after):
    start, end = before["finished_monotonic_s"], after["started_monotonic_s"]
    gap = end - start
    return {
        "model_id": MODEL, "num_prompts": 32, "completed": 32, "failed": 0,
        "start_times": [start + gap / 4] * 32, "ttfts": [gap / 4] * 32,
        "itls": [[gap / 8]] * 32, "duration": gap,
    }


def phase_with_measurement(before, after, model):
    return phases.phase_deltas(before, after, model, measured=phase_measurement(before, after))


def capture_fixture(body, status=200):
    payload = body.encode("utf-8")
    wire = bytearray(
        f"HTTP/1.1 {status} Fixture\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
        + payload
    )
    sock = Mock()
    sock.__enter__ = Mock(return_value=sock)
    sock.__exit__ = Mock(return_value=False)

    def receive(buffer):
        size = min(len(buffer), len(wire))
        buffer[:size] = wire[:size]
        del wire[:size]
        return size

    sock.recv_into.side_effect = receive
    return sock


def proc_fixture(root, *, inode=7319, ticks=502, port=PORT, pid=42):
    uid = os.getuid()
    proc = root / str(pid)
    (proc / "fd").mkdir(parents=True)
    (proc / "ns").mkdir()
    (proc / "ns/net").symlink_to("net:[123]")
    (proc / "fd/3").symlink_to(f"socket:[{inode}]")
    (proc / "stat").write_text(f"{pid} (vllm (frontend)) S " + "0 " * 18 + f"{ticks} 0\n")
    (root / "net").mkdir(exist_ok=True)
    (root / "net/tcp").write_text(
        " sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
        f"  0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 "
        f"00:00000000 00000000 {uid} 0 {inode} 1\n"
    )
    (root / "self/ns").mkdir(parents=True, exist_ok=True)
    (root / "self/ns/net").symlink_to("net:[123]")
    (root / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
    (root / "sys/kernel/random/boot_id").write_text("fixture-boot\n")


class TestPhases(unittest.TestCase):
    def test_capture_retains_body_http_and_actual_receipt_without_process_metric(self):
        before_body = metrics(8)
        after_body = metrics(40)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_fixture(root)
            with (patch.object(phases, "_PROC", root),
                  patch.object(phases.socket, "create_connection",
                               side_effect=[capture_fixture(before_body), capture_fixture(after_body)]) as connect):
                before = phases.capture(PORT, time.monotonic() + 1, MODEL)
                after = phases.capture(PORT, time.monotonic() + 1, MODEL)
        self.assertEqual(before["body"], before_body)
        self.assertEqual(before["http_status"], 200)
        self.assertEqual(connect.call_args.args[0], ("127.0.0.1", PORT))
        self.assertTrue(before["drained"])
        self.assertTrue(before["finished_utc"].endswith("+00:00"))
        self.assertEqual(before["process_identity"]["socket_inode"], 7319)
        self.assertEqual(before["process_identity"]["pid"], 42)
        self.assertEqual(before["process_identity"]["start_ticks"], 502)
        result = phase_with_measurement(before, after, MODEL)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["process_identity_source"], "listener-procfs")
        for delta in result["phases"].values():
            self.assertEqual(delta["count_delta"], 32)
            self.assertEqual(delta["inf_bucket_delta"], 32)
            self.assertEqual(delta["mean_s"], 0.25)
            self.assertEqual(delta["series"][0]["labels"], {"model_name": MODEL, "engine": "0"})
            self.assertEqual(delta["series"][0]["bucket_deltas"]["0.3"], 16)

    def test_process_receipt_detects_change_during_get(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_fixture(root)

            def connect(*args):
                (root / "42/stat").write_text("42 (other) S " + "0 " * 18 + "503 0")
                return capture_fixture(metrics(8, process_start=5))

            with patch.object(phases, "_PROC", root), patch.object(phases.socket, "create_connection", connect):
                result = phases.capture(PORT, time.monotonic() + 1, MODEL)
        self.assertFalse(result["valid"])
        self.assertIn("changed during GET", result["reason"])

    def test_capture_bounded_by_absolute_read_deadline(self):
        # Return bytes frequently enough to satisfy an ordinary idle timeout.
        sock = capture_fixture(metrics(8))
        clock = Mock(return_value=0.0)

        def dribble(buffer):
            clock.return_value += 0.04
            buffer[0] = ord("H")
            return 1

        sock.recv_into.side_effect = dribble
        with (patch.object(phases, "_listener_identity", side_effect=PermissionError("fixture")),
              patch.object(phases.time, "monotonic", clock),
              patch.object(phases.socket, "create_connection", return_value=sock)):
            result = phases.capture(PORT, 0.1, MODEL)
        self.assertFalse(result["valid"])
        self.assertIn("deadline", result["reason"])
        self.assertEqual(sock.recv_into.call_count, 3)

    def test_http_error_and_busy_evidence_are_retained(self):
        for status, body in ((503, "not ready\n"), (200, metrics(8, waiting=1))):
            with (patch.object(phases, "_listener_identity", side_effect=PermissionError("fixture")),
                  patch.object(phases.socket, "create_connection", return_value=capture_fixture(body, status))):
                result = phases.capture(PORT, time.monotonic() + 0.05, MODEL)
            self.assertFalse(result["valid"])
            self.assertEqual(result["body"], body)
            self.assertEqual(result["http_status"], status)

    def test_capture_rejects_drain_from_a_different_model(self):
        body = metrics(10, model="other")
        with (patch.object(phases, "_listener_identity", side_effect=PermissionError("fixture")),
              patch.object(phases.socket, "create_connection", return_value=capture_fixture(body))):
            result = phases.capture(PORT, time.monotonic() + 1, MODEL)
        self.assertFalse(result["valid"])
        self.assertIn("expected model", result["reason"])

    def test_measured_window_is_required_and_bracketed(self):
        before = snapshot(metrics(8, process_start=5), 1)
        after = snapshot(metrics(40, process_start=5), 2)
        measured = phase_measurement(before, after)
        self.assertFalse(phases.phase_deltas(before, after, MODEL)["valid"])
        for shift in (-0.2, 0.3):
            moved = {**measured, "start_times": [x + shift for x in measured["start_times"]]}
            result = phases.phase_deltas(before, after, MODEL, measured=moved)
            self.assertFalse(result["valid"])
            self.assertIn("outside the metrics scrape window", result["reason"])

    def test_other_models_are_not_used_for_count(self):
        def with_other(count, other_count):
            other = metrics(other_count, model="other")
            return metrics(count, process_start=5) + "\n".join(
                line for line in other.splitlines() if not line.startswith("#")
            )
        result = phase_with_measurement(
            snapshot(with_other(16, 64), 1), snapshot(with_other(48, 128), 2), MODEL
        )
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["phases"]["queue"]["count_delta"], 32)

    def test_multiple_engines_require_stable_series_and_aggregate_to_32(self):
        def with_engines(count):
            other = metrics(count, engine=1)
            return metrics(count, process_start=5) + "\n".join(
                line for line in other.splitlines() if not line.startswith("#")
            )
        before = snapshot(with_engines(8), 1)
        after = snapshot(with_engines(24), 2)
        result = phase_with_measurement(before, after, MODEL)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["process_identity_source"], "process_start_time_seconds")
        self.assertEqual([row["count_delta"] for row in result["phases"]["prefill"]["series"]], [16, 16])

    def test_missing_counter_changed_labels_or_process_and_resets_reject_evidence(self):
        left = snapshot(metrics(16, process_start=5), 1)
        body = metrics(48, process_start=5)
        invalid_bodies = {
            "count/+Inf delta": metrics(49, process_start=5),
            "changed": metrics(48, process_start=6),
            "gauge series": metrics(48, engine=1, process_start=5),
            "not drained": metrics(48, waiting=1, process_start=5),
            "seconds": body.replace("queue_time_seconds histogram", "queue_time_seconds gauge"),
            "bucket boundaries": body.replace('le="0.3"', 'le="0.2"'),
            "reset": body.replace(f'le="0.3",engine="0",model_name="{MODEL}"}} 24',
                                  f'le="0.3",engine="0",model_name="{MODEL}"}} 7'),
            "inconsistent": body.replace(f'le="+Inf",engine="0",model_name="{MODEL}"}} 48',
                                         f'le="+Inf",engine="0",model_name="{MODEL}"}} 47'),
            "sum delta disagrees": body.replace(f'request_queue_time_seconds_sum{{engine="0",model_name="{MODEL}"}} 12.0',
                                               f'request_queue_time_seconds_sum{{engine="0",model_name="{MODEL}"}} 120.0'),
            "count/sum labels": "\n".join(
                line for line in body.splitlines() if "decode_time_seconds_count" not in line
            ),
        }
        for reason, invalid in invalid_bodies.items():
            with self.subTest(reason=reason):
                result = phase_with_measurement(left, snapshot(invalid, 2), MODEL)
                self.assertFalse(result["valid"])
                self.assertIn(reason, result["reason"])
        result = phase_with_measurement(snapshot(metrics(16), 1), snapshot(metrics(48), 2), MODEL)
        self.assertFalse(result["valid"])
        self.assertIn("listener process receipt", result["reason"])


def command_args(directory, result_filename="measured.json"):
    return [
        "vllm", "bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
        "--result-dir", directory, "--result-filename", result_filename,
        "--save-result", "--save-detailed", "--num-prompts", "32", "--max-concurrency", "8",
    ]


def detailed_result():
    return {
        "backend": "openai", "num_prompts": 32, "completed": 32, "failed": 0,
        "max_concurrency": 8, "duration": 1.0, "start_times": [101.0 + (i // 8) * 0.2 + (i % 8) * 0.001 for i in range(32)],
        "ttfts": [0.125] * 32, "itls": [[0.01, 0.02]] * 32, "errors": [""] * 32,
        "output_lens": [3] * 32,
    }


def observed_events(measured, origin=100.98):
    recorder = client._Recorder()
    for index, start in enumerate(measured["start_times"]):
        request = SimpleNamespace(request_id=f"vllm-input-{31-index}")
        with patch.object(client.time, "perf_counter", side_effect=[origin + 0.01, start - 0.001]):
            row = recorder.created(origin, request)
            recorder.acquired(row, origin)
        first = start + measured["ttfts"][index] - 0.00001
        recorder.stream(request, start, first, first + sum(measured["itls"][index]))
    return recorder.evidence(measured)


def profile_receipt(events, measured, start_ack, stop_sent, *, stop_ack=None):
    def mark(seconds):
        return {"monotonic_ns": round(seconds * 1e9)}

    def action(sent, ack):
        return {"sent": mark(sent), "ack": mark(ack), "http_status": 200, "error": None}

    return {
        "valid": True, "requires_server_shutdown": False, "client_clock": events["client_clock"],
        "measured_sha256": events["measured_sha256"], "benchmark_duration_s": measured["duration"],
        "benchmark_start_perf_counter_s": events["benchmark_start_perf_counter_s"],
        "start": action(start_ack - 0.005, start_ack),
        "stop": action(stop_sent, stop_ack if stop_ack is not None else stop_sent + 0.005),
    }


class TestClient(unittest.TestCase):
    def test_command_preserves_argument_order_and_refuses_warmup(self):
        argv = command_args("/work/metrics/cell")
        self.assertEqual(client.command(argv), [
            "python3", "-m", "infraestructura.inference_diag_client", *argv[3:],
        ])
        self.assertEqual(client.command(argv, profile_port=PORT), [
            "python3", "-m", "infraestructura.inference_diag_client",
            f"--ipre-profile-port={PORT}", *argv[3:],
        ])
        with self.assertRaises(ValueError):
            client.command(command_args("/work/metrics/cell", "warmup.json"))

    def test_archived_source_instruments_only_benchmark_and_rejects_drift(self):
        archived = (Path(__file__).resolve().parents[1]
                    / "resultados/tmp/inference-runtime-final-20261106/vllm_benchmarks_serve.py")
        if not archived.is_file():
            self.skipTest("local archived image source unavailable")
        source = archived.read_bytes()
        code = client._instrumented_code(source, str(archived))
        self.assertEqual(hashlib.sha256(source).hexdigest(), client.SERVE_SHA256)
        self.assertEqual(code.co_name, "benchmark")
        self.assertIn("_ipre_diag_created", code.co_names)
        self.assertIn("_ipre_diag_before_batch", code.co_names)
        self.assertIn("_ipre_diag_after_batch", code.co_names)
        limited = next(c for c in code.co_consts if getattr(c, "co_name", "") == "limited_request_func")
        self.assertIn("_ipre_diag_acquired", limited.co_names)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            client._instrumented_code(source + b"\n", str(archived))

    def test_verify_client_clock_mapping_and_detect_corruption(self):
        measured = detailed_result()
        evidence = observed_events(measured)
        self.assertEqual(client.verify_events(evidence, measured), {"valid": True, "count": 32})
        self.assertEqual(evidence["rows"][0]["request_id"], "vllm-input-31")
        self.assertAlmostEqual(evidence["rows"][0]["first_token_offset_s"], 0.14499)
        self.assertAlmostEqual(evidence["rows"][0]["last_token_offset_s"], 0.17499)
        self.assertFalse(client.verify_events(evidence, {**measured, "duration": 1.02})["valid"])
        evidence["rows"][0]["last_token_offset_s"] += 0.01
        self.assertFalse(client.verify_events(evidence, measured)["valid"])

    def test_rejects_impossible_c8_and_releases_before_acquiring_at_a_tie(self):
        measured = detailed_result()
        events = observed_events(measured)
        rows = events["rows"]
        rows[8]["semaphore_acquired_offset_s"] = rows[0]["last_token_offset_s"]
        self.assertTrue(client.verify_events(events, measured)["valid"])
        rows[8]["semaphore_acquired_offset_s"] -= 0.0001
        result = client.verify_events(events, measured)
        self.assertFalse(result["valid"])
        self.assertIn("semaphore limit 8", result["reason"])

    def test_profile_coverage_rejects_control_before_first_request(self):
        measured = detailed_result()
        events = observed_events(measured)
        control = profile_receipt(events, measured, 100.9, 101.0)
        result = client.profile_coverage(control, events, measured)
        self.assertEqual(result["count"], 0)
        self.assertFalse(result["valid"])
        self.assertIn("overlaps no measured request", result["reason"])

    def test_profile_coverage_partial_60s_and_mismatched_clock(self):
        measured = detailed_result()
        measured["start_times"] = [101.0 + 20 * (i // 8) + 0.1 * (i % 8) for i in range(32)]
        measured["ttfts"] = [19.0] * 32
        measured["duration"] = 90.0
        events = observed_events(measured)
        control = profile_receipt(events, measured, 100.95, 155.94, stop_ack=160.945)
        self.assertEqual(client.profile_coverage(control, events, measured), {
            "valid": True, "count": 24, "fully_covered_count": 16,
        })
        control["client_clock"] = {**events["client_clock"], "invocation_id": "other"}
        result = client.profile_coverage(control, events, measured)
        self.assertFalse(result["valid"])
        self.assertIn("CLOCK_MONOTONIC", result["reason"])

    def test_profile_wrapper_finalizes_outside_t32_or_on_client_exception(self):
        for interrupted, missing_stop in ((False, False), (False, True), (True, True)):
            with self.subTest(interrupted=interrupted, missing_stop=missing_stop), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "measured.json"
                argv = client.command(command_args(directory), profile_port=PORT)[3:]
                steps = []
                holder = {}

                def install(recorder, _serve):
                    holder["recorder"] = recorder

                @contextmanager
                def window(port, deadline):
                    self.assertEqual(port, PORT)
                    self.assertAlmostEqual(deadline - time.monotonic(), 60, delta=1)
                    receipt = profile_receipt(observed_events(detailed_result()), detailed_result(),
                                              100.9, 102)
                    receipt["errors"] = []
                    stop = receipt.pop("stop")
                    steps.append("start")
                    try:
                        yield receipt
                    finally:
                        steps.append("stop")
                        if not missing_stop:
                            receipt["stop"] = stop

                def original():
                    recorder = holder["recorder"]
                    recorder.before_batch()
                    steps.append("T32 begins")
                    if interrupted:
                        raise RuntimeError("client interrupted")
                    steps.append("T32 ends")
                    recorder.after_batch(100.98, 1.0)
                    path.write_text(json.dumps(detailed_result()))
                    return 0

                with (patch.object(client._Recorder, "install", install),
                      patch.object(client, "import_module"),
                      patch.object(client, "_entry_point", return_value=original),
                      patch.object(client.sys, "argv", []),
                      patch.object(client.sys, "dont_write_bytecode", True),
                      patch("infraestructura.inference_diag_traces.trace_window", window)):
                    if interrupted:
                        with self.assertRaisesRegex(RuntimeError, "client interrupted"):
                            client.main(argv)
                    else:
                        self.assertEqual(client.main(argv), 0)
                control = json.loads(path.with_name(client.PROFILE_FILENAME).read_text())
                self.assertEqual(control["requires_server_shutdown"], missing_stop)
                self.assertEqual(steps, ["start", "T32 begins", *([] if interrupted else ["T32 ends"]), "stop"])
                self.assertEqual(path.exists(), not interrupted)

    def test_interrupted_profile_entry_attempts_emergency_stop(self):
        @contextmanager
        def interrupted(*args, **kwargs):
            raise RuntimeError("profiler entry interrupted")
            yield

        window = client._Recorder(PORT).profile
        with (patch("infraestructura.inference_diag_traces.trace_window", interrupted),
              patch("infraestructura.inference_diag_traces._control") as stop):
            with self.assertRaisesRegex(RuntimeError, "entry interrupted"):
                window.open()
            window.close()
        stop.assert_called_once()
        self.assertEqual(stop.call_args.args[:2], (PORT, "stop"))
        self.assertTrue(window.receipt["requires_server_shutdown"])

    def test_cli_fallback_preserves_completed_original_file_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = command_args(directory)
            path = Path(directory) / "measured.json"
            raw = json.dumps(detailed_result()) + "\n"

            def original():
                self.assertEqual(client.sys.argv, argv)
                path.write_text(raw)
                return 0

            original_main = Mock(side_effect=original)
            with (patch.object(client, "_entry_point", return_value=original_main),
                  patch.object(client, "import_module", side_effect=ValueError("fixture source drift")),
                  patch.object(client.sys, "argv", []),
                  patch.object(client.sys, "dont_write_bytecode", True)):
                status = client.main(argv[3:])
            self.assertEqual(status, 0)
            original_main.assert_called_once_with()
            self.assertEqual(path.read_text(), raw)
            events = json.loads((path.parent / "client-events.json").read_text())
            self.assertIn("fixture source drift", client.verify_events(events, detailed_result())["reason"])


if __name__ == "__main__":
    unittest.main()
