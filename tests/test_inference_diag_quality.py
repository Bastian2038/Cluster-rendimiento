"""CPU fixtures for cell-local verification of diagnostic run evidence."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infraestructura import inference_diag_quality as quality
from infraestructura import inference_observation as observation
from infraestructura import inference_telemetry_report as telemetry


_ORIGIN = datetime(2026, 11, 21, tzinfo=timezone.utc)


def stamp(second):
    return (_ORIGIN + timedelta(seconds=second)).isoformat()


def cell(name, start, end):
    return {"cell": name, "measurement_started_at": stamp(start),
            "measurement_ended_at": stamp(end)}


class CellQualityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ranks = [
            {"node_id": "pegasus", "hostname": "spark-cfea", "rank": 0,
             "gpu_uuid": "GPU-0", "world_size": 2},
            {"node_id": "octopus", "hostname": "spark-60c3", "rank": 1,
             "gpu_uuid": "GPU-1", "world_size": 2},
        ]
        self.cell = cell("2048-c8", 10, 30)
        # Additional log lines with the same signatures belong to other cells or
        # warmup. Byte counts differ from codepoint counts in the Unicode prefix.
        prefix = "Calentamiento: compilación\nStarting compilation\n".encode()
        suffix = b"Capturing CUDA graphs\nFailed to stop profiler\n"
        self.logs = {
            "pegasus": {
                "server.log": [prefix, b"POST /v1/completions HTTP/1.1 200 OK\n", suffix],
                "worker-rank00.log": [prefix, b"", suffix],
            },
            "octopus": {"worker-rank01.log": [prefix, b"", suffix]},
        }
        self.write_logs(self.cell)
        for node in self.logs:
            self.write_telemetry(node, (0, 10, 14, 18, 22, 26, 30, 45))

    def log_dir(self, node):
        return self.root / "raw/nodes" / node / "logs"

    def receipt_path(self, record):
        return self.root / "raw/diagnostic" / record["cell"] / "execution-window.json"

    def write_logs(self, record):
        receipt = record | {"logs": {}}
        for node, logs in self.logs.items():
            receipt["logs"][node] = {}
            for name, (before, measured, after) in logs.items():
                filename = self.log_dir(node) / name
                filename.parent.mkdir(parents=True, exist_ok=True)
                filename.write_bytes(before + measured + after)
                receipt["logs"][node][name] = [len(before), len(before) + len(measured)]
        path = self.receipt_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")

    def write_telemetry(self, node, seconds, *, invalid_time=None, invalid_gpu=None):
        rank = next(worker for worker in self.ranks if worker["node_id"] == node)
        proc = {"MemAvailable_kib": 22 * 1024 * 1024, "MemTotal_kib": 120 * 1024 * 1024}
        # Reuse the real sampler's record shape with CPU-only fixtures for the
        # sources. GB10 GPU memory/power may be unavailable in archived samples.
        with patch.object(observation.socket, "gethostname", return_value=rank["hostname"]), \
                patch.object(observation, "_meminfo", return_value=proc), \
                patch.object(observation, "_pressure", return_value={}), \
                patch.object(observation, "_cpu", return_value={}), \
                patch.object(observation, "_nic", return_value={}), \
                patch.object(observation, "_gpu_data", return_value=[
                    {"uuid": rank["gpu_uuid"], "memory_used": None, "memory_total": None,
                     "power_draw": None},
                ]), patch.object(observation, "_gpu_apps", return_value=[]):
            template = observation.collect_sample()
        records = []
        for second in seconds:
            sample = template | {"timestamp_utc": stamp(second), "monotonic_ns": second * 10**9}
            if second == invalid_time:
                sample["monotonic_ns"] = 0
            if second == invalid_gpu:
                sample["gpus"] = [{"uuid": "GPU-unexpected"}]
            records.append(json.dumps(sample))
        (self.log_dir(node) / "telemetry.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")

    def issues(self):
        return quality.cell_issues(self.root, self.cell, self.ranks)

    def test_clean_cell_accepts_silent_workers_and_unavailable_optional_gpu_metrics(self):
        self.assertEqual(self.issues(), [])

    def test_worker_compilation_affects_only_its_cell(self):
        initial, _, final = self.logs["octopus"]["worker-rank01.log"]
        self.logs["octopus"]["worker-rank01.log"] = [
            initial, b"Compiling a CUDAGraph dependency\n", final,
        ]
        self.write_logs(self.cell)
        issues = self.issues()
        self.assertEqual(len(issues), 1, issues)
        self.assertIn("octopus/worker-rank01.log: compilation", issues[0])

        clean = cell("512-c1", 32, 44)
        # Archive the same affected bytes in the run, with a silent worker
        # interval for the other cell. Its receipt points past the compilation.
        for node in self.logs:
            self.write_telemetry(node, (0, 10, 14, 18, 22, 26, 30, 34, 38, 42, 45))
        clean_file = self.receipt_path(clean)
        receipt = json.loads(self.receipt_path(self.cell).read_text()) | clean
        cutoff = len(initial) + len(b"Compiling a CUDAGraph dependency\n")
        for node, logs in receipt["logs"].items():
            for name, offsets in logs.items():
                logs[name] = [offsets[1], offsets[1]]
        receipt["logs"]["octopus"]["worker-rank01.log"] = [cutoff, cutoff]
        clean_file.parent.mkdir(parents=True)
        clean_file.write_text(json.dumps(receipt))
        self.assertEqual(quality.cell_issues(self.root, clean, self.ranks), [])

    def test_absence_and_gap_are_local_to_the_participant(self):
        self.write_telemetry("octopus", (0, 45))
        self.assertIn("no samples inside", self.issues()[0])
        self.write_telemetry("octopus", (0, 10, 24, 30, 45))
        issues = self.issues()
        self.assertEqual(len(issues), 1, issues)
        self.assertIn("octopus/telemetry.jsonl", issues[0])
        self.assertIn("gap exceeds 10 s", issues[0])

    def test_cell_requires_coverage_at_both_edges(self):
        self.write_telemetry("pegasus", (0, 14, 20, 26, 45))
        self.assertIn("cover measured window edges", self.issues()[0])

    def test_wrong_gpu_outside_cell_is_ignored_but_inside_fails(self):
        seconds = (0, 10, 14, 18, 22, 26, 30, 45)
        self.write_telemetry("octopus", seconds, invalid_gpu=45)
        self.assertEqual(self.issues(), [])
        self.write_telemetry("octopus", seconds, invalid_gpu=18)
        self.assertIn("GPU UUID", self.issues()[0])

    def test_nonmonotonic_telemetry_in_cell_fails(self):
        self.write_telemetry("octopus", (10, 14, 18, 22, 26, 30), invalid_time=18)
        self.assertIn("non-monotonic", self.issues()[0])

    def test_truncated_tail_after_first_later_sample_does_not_affect_cell(self):
        self.write_telemetry("octopus", (0, 10, 14, 18, 22, 26, 30, 93))
        path = self.log_dir("octopus") / "telemetry.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"timestamp_utc":"\xff')
        start, end = (_ORIGIN + timedelta(seconds=value) for value in (10, 30))
        samples = quality.telemetry_window(path, start, end, self.ranks[1])
        self.assertEqual([s["_utc_seconds"] - _ORIGIN.timestamp() for s in samples],
                         [10, 14, 18, 22, 26, 30])
        self.assertEqual(telemetry.summarize(samples, start.timestamp(), end.timestamp())["sample_count"], 6)
        self.assertEqual(self.issues(), [])

    def test_unreadable_row_between_measured_samples_excludes_cell(self):
        path = self.log_dir("octopus") / "telemetry.jsonl"
        rows = path.read_bytes().splitlines(keepends=True)
        rows.insert(3, b'{"malformed"\n')
        path.write_bytes(b"".join(rows))
        self.assertIn("unreadable telemetry overlaps measured window", self.issues()[0])

    def test_unreadable_prefix_isolated_by_later_samples(self):
        path = self.log_dir("octopus") / "telemetry.jsonl"
        path.write_bytes(b'{"malformed"\n' + path.read_bytes())
        self.assertEqual(self.issues(), [])
        self.write_telemetry("octopus", (12, 16, 20, 24, 28, 45))
        path.write_bytes(b'{"malformed"\n' + path.read_bytes())
        self.assertIn("unreadable telemetry overlaps measured window", self.issues()[0])

    def test_unreadable_row_before_right_bracket_protects_measured_edge(self):
        self.write_telemetry("octopus", (10, 14, 18, 22, 26, 29, 45))
        path = self.log_dir("octopus") / "telemetry.jsonl"
        rows = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(rows[:-1]) + b'{"malformed"\n' + rows[-1])
        self.assertIn("unreadable telemetry overlaps measured window", self.issues()[0])

    def test_utc_disorder_before_window_is_local_but_disorder_at_edge_fails(self):
        self.write_telemetry("octopus", (3, 2, 10, 14, 18, 22, 26, 30, 45))
        self.assertEqual(self.issues(), [])
        self.write_telemetry("octopus", (3, 10, 9, 14, 18, 22, 26, 30, 45))
        self.assertIn("non-monotonic telemetry", self.issues()[0])

    def test_profiler_failure_detected_in_exact_server_window(self):
        self.logs["pegasus"]["server.log"][1] = b"WARNING Failed to stop profiler: unavailable\n"
        self.write_logs(self.cell)
        self.assertIn("server.log: Failed to stop profiler", self.issues()[0])
        self.assertTrue(quality.profiler_stop_failed("WARNING Failed to stop profiler."))
        self.assertFalse(quality.profiler_stop_failed("Profiler stopped."))

    def test_server_graph_capture_is_compilation_evidence(self):
        self.logs["pegasus"]["server.log"][1] = b"Capturing CUDA graphs (decode)\n"
        self.write_logs(self.cell)
        self.assertIn("server.log: compilation or CUDA graph capture", self.issues()[0])

    def test_missing_sealed_worker_and_bad_receipt_are_distinct(self):
        (self.log_dir("octopus") / "worker-rank01.log").unlink()
        self.assertIn("evidencia ausente", self.issues()[0])
        receipt = self.receipt_path(self.cell)
        data = json.loads(receipt.read_text())
        data["measurement_started_at"] = stamp(9)
        receipt.write_text(json.dumps(data))
        self.assertIn("another cell or measured interval", self.issues()[0])

    def test_wrong_or_missing_offsets_fail_closed(self):
        receipt = self.receipt_path(self.cell)
        data = json.loads(receipt.read_text())
        data["logs"]["pegasus"]["server.log"][0] += 1
        receipt.write_text(json.dumps(data))
        self.assertIn("cuts a line", self.issues()[0])
        del data["logs"]["pegasus"]["server.log"]
        receipt.write_text(json.dumps(data))
        self.assertIn("log set differs", self.issues()[0])
        receipt.unlink()
        self.assertIn("execution-window", self.issues()[0])

    def test_subset_of_ranks_does_not_attest_complete_world(self):
        issues = quality.cell_issues(self.root, self.cell, self.ranks[:1])
        self.assertIn("incomplete world", issues[0])


if __name__ == "__main__":
    unittest.main()
