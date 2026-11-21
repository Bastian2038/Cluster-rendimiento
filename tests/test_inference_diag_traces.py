import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from infraestructura import inference_diag_traces as traces


def event(category, name, ts, duration, *, pid=71, tid=72, **args):
    return {"ph": "X", "cat": category, "name": name, "ts": ts, "dur": duration,
            "pid": pid, "tid": tid, "args": args}


def kernel(name, ts, duration, stream, **args):
    return event("kernel", name, ts, duration, pid=0, tid=stream,
                 device=0, stream=stream, **args)


def kineto(rank):
    return {
        "schemaVersion": 1,
        "deviceProperties": [{"id": 0, "name": "NVIDIA GB10"}],
        "distributedInfo": {"backend": "nccl", "rank": rank, "world_size": 2},
        "traceEvents": [
            {"ph": "M", "name": "process_name", "args": {"name": "python"}},
            event("user_annotation", "execute_context_1(512)_generation_0(0)", 0, 10),
            event("user_annotation", "execute_context_0(0)_generation_2(2)", 10, 10),
            event("user_annotation", "execute_context_1(32)_generation_2(2)", 20, 10),
            event("cuda_runtime", "cudaLaunchKernel", 1, 1, correlation=10),
            event("cuda_runtime", "cudaLaunchKernel", 12, 1, correlation=11),
            event("cuda_driver", "cuLaunchKernelEx", 21, 1, correlation=12),
            event("cpu_op", "aten::mm", 3, 2, **{"External id": 24}),
            kernel("gemm", 100, 12, 7, correlation=10),
            kernel("cutlass::gemm", 104, 10, 8, **{"External id": 24}),
            kernel("ncclDevKernel_AllReduce", 106, 4, 7, correlation=11),
            kernel("ncclKernel_AllGather", 108, 8, 9, correlation=12),
            kernel("unlabeled", 200, 2, 9),
            event("gpu_memcpy", "Memcpy DtoD", 100, 500),
        ],
    }


class TestTraceWindow(unittest.TestCase):
    def test_server_args(self):
        flag, payload = traces.server_args()
        config = json.loads(payload)
        self.assertEqual(flag, "--profiler-config")
        self.assertEqual(config["profiler"], "torch")
        self.assertEqual(config["torch_profiler_dir"], "/work/traces")
        self.assertTrue(config["torch_profiler_use_gzip"])
        self.assertTrue(config["ignore_frontend"])
        self.assertEqual(config["max_iterations"], 32)
        for option in ("with_stack", "record_shapes", "with_memory", "with_flops",
                       "dump_cuda_time_total"):
            self.assertIs(config[f"torch_profiler_{option}"], False)

    def test_client_failure_still_flushes_and_propagates(self):
        conn = Mock()
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=False)
        conn.recv.return_value = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        with patch.object(traces.socket, "create_connection", return_value=conn) as connect:
            with self.assertRaisesRegex(RuntimeError, "client failed"):
                with traces.trace_window(9000, time.monotonic() + 4) as receipt:
                    self.assertIsNotNone(receipt["start"]["ack"])
                    raise RuntimeError("client failed")
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["stop_reason"], "client_exit")
        self.assertIsNotNone(receipt["stop"]["ack"])
        self.assertTrue(receipt["covered_client"])
        self.assertGreaterEqual(receipt["outer_duration_seconds"], receipt["observed_window_seconds"])
        self.assertGreaterEqual(receipt["observed_window_seconds"], receipt["client_duration_seconds"])
        self.assertEqual([call.args[0] for call in connect.call_args_list],
                         [("127.0.0.1", 9000), ("127.0.0.1", 9000)])
        self.assertEqual([call.args[0].split()[1] for call in conn.sendall.call_args_list],
                         [b"/start_profile", b"/stop_profile"])

    def test_start_timeout_and_http_stop_error_leave_marks(self):
        conn = Mock()
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=False)
        conn.recv.return_value = b"HTTP/1.1 500 Internal Server Error\r\n\r\n"
        with patch.object(traces.socket, "create_connection", side_effect=[TimeoutError("start expired"), conn]):
            with traces.trace_window(9000, time.monotonic() + 4) as receipt:
                self.assertIsNone(receipt["start"]["ack"])
        self.assertFalse(receipt["valid"])
        self.assertFalse(receipt["covered_client"])
        self.assertIsNone(receipt["observed_window_seconds"])
        self.assertEqual(receipt["stop_reason"], "start_failed")
        self.assertIn("start expired", receipt["errors"][0])
        self.assertIn("HTTP 500", receipt["errors"][1])
        self.assertEqual(receipt["stop"]["http_status"], 500)
        self.assertIsNotNone(receipt["stop"]["sent"])
        self.assertIsNotNone(receipt["stop"]["finished"])

    def test_stop_watchdog_runs_while_client_is_blocked(self):
        stop_called = threading.Event()

        def control(_port, action, deadline, receipt):
            receipt[action] = {"sent": traces._mark(), "ack": traces._mark(),
                               "error": None, "deadline": deadline}
            if action == "stop":
                stop_called.set()

        with patch.object(traces, "MAX_WINDOW_SECONDS", 0.4), patch.object(traces, "_control", control):
            with traces.trace_window(9000, time.monotonic() + 3) as receipt:
                self.assertTrue(stop_called.wait(1))
                self.assertIn("stop", receipt)
                self.assertLessEqual(receipt["stop"]["deadline"], receipt["hard_deadline_monotonic_s"])
                # Model a client that continues even beyond the hard profiler ceiling.
                time.sleep(max(0, receipt["hard_deadline_monotonic_s"] - time.monotonic()) + 0.01)
        self.assertEqual(receipt["stop_reason"], "deadline")
        self.assertTrue(receipt["valid"])
        self.assertFalse(receipt["covered_client"])
        self.assertEqual(receipt["observed_window_seconds"],
                         (receipt["stop"]["ack"]["monotonic_ns"]
                          - receipt["start"]["sent"]["monotonic_ns"]) / 1e9)
        self.assertLessEqual(receipt["observed_window_seconds"], 0.4)
        self.assertGreater(receipt["client_duration_seconds"], receipt["observed_window_seconds"])
        self.assertGreater(receipt["outer_duration_seconds"], 0.4)

    def test_stop_timeout_after_successful_start_invalidates_window(self):
        conn = Mock()
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=False)
        conn.recv.side_effect = [b"HTTP/1.1 200 OK\r\n\r\n", TimeoutError("stop expired")]
        with patch.object(traces.socket, "create_connection", return_value=conn):
            with traces.trace_window(9000, time.monotonic() + 4) as receipt:
                pass
        self.assertIsNotNone(receipt["start"]["ack"])
        self.assertIsNotNone(receipt["stop"]["sent"])
        self.assertIsNone(receipt["stop"]["ack"])
        self.assertIsNone(receipt["observed_window_seconds"])
        self.assertFalse(receipt["covered_client"])
        self.assertFalse(receipt["valid"])

    def test_ack_after_hard_deadline_keeps_observed_interval_but_invalidates(self):
        def control(_port, action, _deadline, receipt):
            sent, ack = traces._mark(), traces._mark()
            if action == "stop":
                ack["monotonic_ns"] = int(receipt["hard_deadline_monotonic_s"] * 1e9) + 10_000
            receipt[action] = {"sent": sent, "ack": ack, "error": None}

        with patch.object(traces, "_control", control):
            with traces.trace_window(9000, time.monotonic() + 4) as receipt:
                pass
        self.assertIsNotNone(receipt["observed_window_seconds"])
        self.assertIn("exceeds its deadline", receipt["errors"][0])
        self.assertFalse(receipt["covered_client"])
        self.assertFalse(receipt["valid"])

    def test_already_expired_has_both_attempts_without_connection(self):
        with patch.object(traces.socket, "create_connection") as connect:
            with traces.trace_window(9000, time.monotonic() - 1) as receipt:
                pass
        connect.assert_not_called()
        self.assertFalse(receipt["valid"])
        for action in ("start", "stop"):
            self.assertIn("deadline", receipt[action]["error"])
            self.assertIn("attempt", receipt[action])


class TestTraceCollection(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.nodes = [{"node_id": "pegasus"}, {"node_id": "octopus"}]
        self.ranks = [{"node_id": node["node_id"], "hostname": node["node_id"] + "-host",
                       "gpu_uuid": f"GPU-fixture-{rank}", "rank": rank, "device": "cuda:0",
                       "world_size": 2} for rank, node in enumerate(self.nodes)]
        self.seen = {}
        remote = patch.object(traces.runtime, "remote", side_effect=self.local_remote)
        self.remote = remote.start()
        self.addCleanup(remote.stop)

    def local_remote(self, node, argv, *, timeout):
        # Run the production scanner with private CPU fixtures, using no SSH.
        argv = [sys.executable, "-B", *argv[1:3], str(self.root / node["node_id"]), *argv[4:]]
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def fixture(self, rank, capture=1, *, name=None, data=None, contents=None):
        directory = self.root / self.ranks[rank]["node_id"] / "traces"
        directory.mkdir(parents=True, exist_ok=True)
        name = name or f"dp0_pp0_tp{rank}_dcp0_ep0_rank{rank}.{capture}.pt.trace.json.gz"
        path = directory / name
        path.write_bytes(contents if contents is not None else
                         gzip.compress(json.dumps(data if data is not None else kineto(rank)).encode()))
        return path

    def collect(self, cell, deadline=3):
        dest = self.root / "run" / "raw" / "diagnostic" / cell / "traces"
        receipt = traces.collect_traces(self.nodes, "/isolated/campaign/job", dest, self.ranks,
                                        self.seen, time.monotonic() + deadline)
        return dest, receipt

    def test_rank_stream_unions_and_asynchronous_correlation_are_reproducible(self):
        self.fixture(0)
        self.fixture(1)
        dest, receipt = self.collect("cell1")
        self.assertTrue(receipt["valid"], msg=receipt)
        self.assertEqual(traces.validate_traces(dest, receipt, reversed(self.ranks)), receipt)
        rank, other = receipt["summary"]
        self.assertEqual(rank["rank"], 0)
        self.assertEqual(other["rank"], 1)
        self.assertEqual(rank["activity"], {"compute_us": 16, "nccl_us": 10, "overlap_us": 8})
        self.assertEqual(rank["stream_activity"], [
            {"stream": 7, "compute_us": 12, "nccl_us": 4, "overlap_us": 4},
            {"stream": 8, "compute_us": 10, "nccl_us": 0, "overlap_us": 0},
            {"stream": 9, "compute_us": 2, "nccl_us": 8, "overlap_us": 0},
        ])
        self.assertEqual([row["category"] for row in rank["cpu_iteration_ranges"]],
                         ["context", "decode", "mixed"])
        self.assertEqual({row["category"]: row["kernel_count"] for row in rank["iteration_activity"]},
                         {"context": 2, "decode": 1, "mixed": 1, "insufficient": 1})
        self.assertTrue(all(row["sha256"] for row in receipt["files"]))

        self.fixture(0, 2)
        self.fixture(1, 2)
        _, second = self.collect("cell2")
        self.assertTrue(second["valid"], msg=second)
        self.assertTrue(all(".2.pt.trace" in row["path"] for row in second["files"]))
        self.assertEqual([len(value) for value in self.seen.values()], [2, 2])
        self.assertEqual(traces.validate_traces(dest, receipt, self.ranks), receipt)

    def test_missing_capture_quarantines_late_exports(self):
        self.fixture(0)
        self.assertFalse(self.collect("incomplete", deadline=0.25)[1]["valid"])
        self.fixture(1)
        self.remote.reset_mock()
        dest, failed = self.collect("next")
        self.assertFalse(failed["valid"])
        self.assertIn("incomplete capture", failed["errors"][0])
        self.assertFalse(dest.exists())
        self.remote.assert_not_called()

    def test_wrong_rank_host_or_device_is_rejected(self):
        for field, value in (("distributedInfo", {"rank": 0}), ("hostname", "unexpected"),
                             ("deviceProperties", [{"id": 0, "uuid": "GPU-wrong"}])):
            with self.subTest(field=field):
                self.seen.clear()
                self.fixture(0)
                data = kineto(1)
                data[field] = value
                self.fixture(1, data=data)
                dest, receipt = self.collect("identity")
                self.assertFalse(receipt["valid"])
                self.assertFalse(dest.exists())
        self.seen.clear()
        data = kineto(1)
        data["traceEvents"][8]["args"]["device"] = 1
        self.fixture(1, data=data)
        self.assertFalse(self.collect("wrong-device")[1]["valid"])

    def test_local_rank_is_not_global_evidence(self):
        data = kineto(1)
        del data["distributedInfo"]
        self.fixture(0)
        self.fixture(1, name="local_rank0_tp_rank1.1.pt.trace.json.gz", data=data)
        _, receipt = self.collect("local-only")
        self.assertFalse(receipt["valid"])
        self.assertIn("global rank", receipt["errors"][0])

    def test_image_worker_suffix_and_fallback_identify_global_rank(self):
        for rank, name in enumerate(("dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json.gz",
                                     "rank1.1.pt.trace.json.gz")):
            data = kineto(rank)
            del data["distributedInfo"]
            self.fixture(rank, name=name, data=data)
        _, receipt = self.collect("worker-suffix")
        self.assertTrue(receipt["valid"], msg=receipt)
        self.assertEqual([row["identity_evidence"]["global_rank"] for row in receipt["summary"]],
                         ["filename", "filename"])

    def test_metadata_alone_reports_gpu_characteristics_without_phase_guess(self):
        data = kineto(1)
        data["traceEvents"] = [kernel("graph_generated_kernel", 3, 7, 8)]
        data["hostname"] = self.ranks[1]["hostname"]
        data["deviceProperties"][0]["uuid"] = self.ranks[1]["gpu_uuid"]
        self.fixture(0)
        self.fixture(1, name="worker.1.pt.trace.json.gz", data=data)
        _, receipt = self.collect("no-label")
        self.assertTrue(receipt["valid"], msg=receipt)
        summary = receipt["summary"][1]
        self.assertEqual(summary["iteration_status"], "insufficient")
        self.assertEqual(summary["nccl_label_status"], "insufficient")
        self.assertEqual(summary["cpu_iteration_ranges"], [])
        self.assertEqual(summary["activity"]["compute_us"], 7)
        self.assertEqual(summary["identity_evidence"]["gpu_uuid"], "trace_metadata_and_worker")

    def test_conflicting_correlation_does_not_label_phase(self):
        data = kineto(1)
        data["traceEvents"].append(event("cuda_runtime", "cudaLaunchKernel", 16, 1,
                                         correlation=10))
        self.fixture(0)
        self.fixture(1, data=data)
        _, receipt = self.collect("conflict")
        self.assertTrue(receipt["valid"], msg=receipt)
        counts = {row["category"]: row["kernel_count"]
                  for row in receipt["summary"][1]["iteration_activity"]}
        self.assertEqual(counts["context"], 1)
        self.assertEqual(counts["insufficient"], 2)

    def test_external_id_does_not_override_ambiguous_launch(self):
        data = kineto(1)
        data["traceEvents"][8]["args"]["External id"] = 24
        data["traceEvents"].append(event("cuda_runtime", "cudaLaunchKernel", 40, 1,
                                         correlation=10))
        self.fixture(0)
        self.fixture(1, data=data)
        _, receipt = self.collect("unannotated-launch")
        self.assertTrue(receipt["valid"], msg=receipt)
        counts = {row["category"]: row["kernel_count"]
                  for row in receipt["summary"][1]["iteration_activity"]}
        self.assertEqual(counts["insufficient"], 2)

    def test_other_collective_and_missing_stream_keep_uncertainty(self):
        data = kineto(1)
        data["traceEvents"] = [
            kernel("vllm::customAllReduce", 1, 8, 8),
            event("kernel", "gemm", 4, 10, pid=0, device=0),
        ]
        self.fixture(0)
        self.fixture(1, data=data)
        _, receipt = self.collect("unresolved-kernel")
        self.assertTrue(receipt["valid"], msg=receipt)
        summary = receipt["summary"][1]
        self.assertEqual(summary["unclassified_kernel_us"], 8)
        self.assertEqual(summary["unresolved_stream_kernel_count"], 1)
        self.assertEqual(summary["activity"], {"compute_us": 10, "nccl_us": 0, "overlap_us": 0})
        self.assertEqual(summary["stream_activity"], [])

    def test_extra_or_replaced_remote_file_invalidates_capture(self):
        known = self.fixture(0)
        self.fixture(1)
        self.assertTrue(self.collect("baseline")[1]["valid"])
        known.write_bytes(known.read_bytes() + b"changed")
        self.fixture(0, 2)
        self.fixture(1, 2)
        self.assertIn("changed", self.collect("changed")[1]["errors"][0])
        self.seen.clear()
        self.assertIn("multiple new", self.collect("extra")[1]["errors"][0])

    def test_tampered_local_contents_path_and_symlink_fail_revalidation(self):
        self.fixture(0)
        self.fixture(1)
        dest, receipt = self.collect("verified")
        self.assertTrue(receipt["valid"])
        rel = receipt["files"][0]["path"]
        file = dest / rel
        content = file.read_bytes()
        file.write_bytes(b"x" + content[1:])
        self.assertIn("SHA-256", traces.validate_traces(dest, receipt, self.ranks)["errors"][0])
        file.unlink()
        file.symlink_to(self.root / rel)
        self.assertFalse(traces.validate_traces(dest, receipt, self.ranks)["valid"])
        receipt["files"][0]["path"] = f"../{rel}"
        self.assertFalse(traces.validate_traces(dest, receipt, self.ranks)["valid"])

    def test_rejects_truncated_gzip_and_bounded_expansion(self):
        self.fixture(0)
        self.fixture(1, contents=gzip.compress(json.dumps(kineto(1)).encode())[:-5])
        self.assertFalse(self.collect("truncated")[1]["valid"])
        self.seen.clear()
        self.fixture(1)
        with patch.object(traces, "MAX_UNCOMPRESSED_BYTES", 128):
            self.assertIn("decompressed", self.collect("expansion")[1]["errors"][0])

    def test_rejects_remote_symlink(self):
        source = self.fixture(0)
        second = self.fixture(1)
        second.unlink()
        second.symlink_to(source)
        dest, receipt = self.collect("link")
        self.assertFalse(receipt["valid"])
        self.assertFalse(dest.exists())
        self.assertIn("regular", receipt["errors"][0])

    def test_compressed_limit_and_per_start_limit(self):
        self.fixture(0)
        self.fixture(1)
        with patch.object(traces, "MAX_COMPRESSED_BYTES", 32):
            self.assertIn("compressed", self.collect("oversized")[1]["errors"][0])
        self.seen.clear()
        with patch.object(traces, "MAX_FILES_PER_NODE", 1):
            self.assertTrue(self.collect("first")[1]["valid"])
            self.fixture(0, 2)
            self.fixture(1, 2)
            self.assertIn("per-start", self.collect("limited")[1]["errors"][0])


if __name__ == "__main__":
    unittest.main()
