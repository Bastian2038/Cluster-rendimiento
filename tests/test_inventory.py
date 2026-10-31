"""Pruebas sintéticas: ninguna consulta del host, carga GPU ni espera real."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infraestructura import inventory, manifest


def config():
    return {
        "node": {"alias": "Pegasus", "expected_hostname": "test-pegasus",
                 "node_count": 1, "role": "observer", "rank": None},
        "duration_seconds": 60, "interval_seconds": 30,
        "warmup_seconds": 0, "trial": 1, "repetitions": 1,
        "idle_thresholds": {"cpu_busy_p95_pct": 10,
                            "gpu_utilization_p95_pct": 5,
                            "memory_available_drift_mib": 1024,
                            "max_sample_lateness_seconds": .5},
        "notes": ["Prueba con datos sintéticos."],
    }


def result(stdout="", status="ok"):
    return {"status": status, "stdout": stdout}


def sample():
    return {"cpu_busy_pct": 2, "memory": {"MemAvailable": 100000000,
            "MemTotal": 127600748, "SwapTotal": 16777212, "SwapFree": 16777212},
            "gpus": [{"uuid": "GPU-test", "utilization_pct": 0}],
            "gpu_processes": [], "lateness_seconds": 0,
            "collection_seconds": .01, "interval_seconds": 30}


class InventoryParserTests(unittest.TestCase):
    def test_meminfo_and_vmstat(self):
        self.assertEqual(inventory.parse_meminfo(
            "MemTotal: 127600748 kB\nMemAvailable: 122695200 kB\n\n"
        ), {"MemTotal": 127600748, "MemAvailable": 122695200})
        self.assertEqual(inventory.parse_meminfo("pswpin 3\npswpout 5\n"),
                         {"pswpin": 3, "pswpout": 5})

    def test_cpu_guest_not_double_counted_and_iowait_separate(self):
        counters = inventory.cpu_counters("cpu  10 2 3 80 5 0 0 0 10 2\ncpu0 1\n")
        self.assertEqual(counters, [10, 2, 3, 80, 5, 0, 0, 0])
        self.assertEqual(inventory.cpu_delta([0] * 8, counters), (15, 5))
        self.assertEqual(inventory.cpu_delta(counters, counters), (None, None))
        self.assertEqual(inventory.cpu_delta(counters, [0] * 8), (None, None))
        with self.assertRaises(ValueError):
            inventory.cpu_counters("cpu 1 2 3")

    def test_gpu_unified_memory_na_is_not_zero(self):
        gpu = inventory.parse_gpu(result(
            "GPU-test, 0, [N/A], N/A, 40, 8.5, 300, P8\n"
        ))[0]
        self.assertEqual(gpu["utilization_pct"], 0)
        self.assertIsNone(gpu["memory_total_mib"])
        self.assertIsNone(gpu["memory_used_mib"])
        self.assertEqual(gpu["availability"]["memory_used_mib"],
                         "unsupported_or_unavailable")
        self.assertEqual(gpu["temperature_c"], 40)
        self.assertEqual(gpu["power_w"], 8.5)
        self.assertEqual(gpu["pstate"], "P8")
        self.assertEqual(inventory.parse_gpu(result(status="timeout")), [])

    def test_gpu_malformed_and_nonfinite(self):
        with self.assertRaises(ValueError):
            inventory.parse_gpu(result("GPU-test, 2\n"))
        row = inventory.parse_gpu(result("GPU-test,nan,N/A,N/A,40,inf,300,P8"))[0]
        self.assertIsNone(row["utilization_pct"])
        self.assertEqual(row["availability"]["power_w"], "invalid")

    def test_process_visibility_empty_present_and_unavailable(self):
        self.assertEqual(inventory.gpu_processes(result()), [])
        self.assertEqual(inventory.gpu_processes(result("GPU-test, 123, VLLM::EngineCore\n")),
                         [{"uuid": "GPU-test", "pid": 123, "name": "VLLM::EngineCore"}])
        self.assertIsNone(inventory.gpu_processes(result(status="error")))
        self.assertIsNone(inventory.gpu_processes(result("Not Supported")))

    def test_probe_records_failures_without_shell(self):
        with mock.patch.object(inventory.subprocess, "run", side_effect=FileNotFoundError("missing")) as run:
            observed = inventory.probe(["missing-tool"])
        self.assertEqual(observed["status"], "unavailable")
        self.assertEqual(observed["command"], ["missing-tool"])
        self.assertNotIn("shell", run.call_args.kwargs)


class InventorySummaryTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.samples = [sample(), sample()]
        self.inventory = {"probes": {name: result() for name in inventory.COMMANDS}}

    def summary(self):
        return inventory.summarize(self.samples, self.config, self.inventory)

    def test_complete_evidence_validates_idle_despite_optional_na(self):
        summary = self.summary()
        self.assertEqual(summary["idle_status"], "validated")
        self.assertEqual(summary["missing_or_invalid"], [])
        self.assertEqual(summary["optional_gpu_metrics"]["power_w"]["count"], 0)

    def test_busy_cpu_and_compute_process_are_not_idle(self):
        self.samples[0]["cpu_busy_pct"] = 80
        self.samples[1]["gpu_processes"] = [{"pid": 1}]
        summary = self.summary()
        self.assertEqual(summary["idle_status"], "not_idle")
        self.assertIn("cpu_busy_p95", summary["violations"])
        self.assertIn("gpu_compute_process_present", summary["violations"])

    def test_missing_visibility_gpu_and_inventory_are_unverified(self):
        self.samples[0]["gpu_processes"] = None
        self.samples[1]["gpus"] = []
        self.inventory["probes"]["services"]["status"] = "error"
        summary = self.summary()
        self.assertEqual(summary["idle_status"], "unverified")
        for reason in ("gpu_process_visibility", "gpu", "expected_single_gpu", "inventory_services"):
            self.assertIn(reason, summary["missing_or_invalid"])

    def test_timing_and_sample_count_are_required_evidence(self):
        self.samples.pop()
        self.samples[0]["lateness_seconds"] = 2
        summary = self.summary()
        self.assertEqual(summary["idle_status"], "unverified")
        self.assertIn("sample_count", summary["missing_or_invalid"])
        self.assertIn("sampling_schedule_out_of_tolerance", summary["missing_or_invalid"])

    def test_memory_drift_is_detected(self):
        self.samples[-1]["memory"]["MemAvailable"] -= 2048 * 1024
        self.assertIn("memory_available_drift", self.summary()["violations"])

    def test_config_rejects_invalid_protocol_values(self):
        inventory.validate_config(self.config)
        for field, value in (("duration_seconds", 0), ("duration_seconds", 30),
                             ("duration_seconds", float("nan")), ("duration_seconds", True),
                             ("interval_seconds", .5), ("interval_seconds", 7),
                             ("warmup_seconds", 5), ("trial", 2), ("repetitions", 2)):
            with self.subTest(field=field, value=value):
                altered = {**self.config, field: value}
                with self.assertRaises(ValueError):
                    inventory.validate_config(altered)
        for key, value in (("alias", ""), ("expected_hostname", ""),
                           ("node_count", 2), ("node_count", 4)):
            altered = copy.deepcopy(self.config)
            altered["node"][key] = value
            with self.assertRaises(ValueError):
                inventory.validate_config(altered)
        altered = copy.deepcopy(self.config)
        altered["node"]["alias"] = "Octopus"
        inventory.validate_config(altered)
        for value in (-1, True, float("inf")):
            altered = copy.deepcopy(self.config)
            altered["idle_thresholds"]["cpu_busy_p95_pct"] = value
            with self.assertRaises(ValueError):
                inventory.validate_config(altered)


class InventoryCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config()))
        self.clock = 0
        self.cpu_reads = 0

    def capture(self, *, missing_gpu=False, broken_cpu=False, host="test-pegasus"):
        read_text = Path.read_text

        def fake_read(path, *args, **kwargs):
            if str(path) == "/proc/stat":
                if broken_cpu:
                    raise OSError("synthetic proc failure")
                self.cpu_reads += 1
                return f"cpu 1 0 0 {100 * self.cpu_reads} 0 0 0 0\n"
            if str(path) == "/proc/meminfo":
                return ("MemTotal: 127600748 kB\nMemAvailable: 122695200 kB\n"
                        "SwapTotal: 16777212 kB\nSwapFree: 16777212 kB\n")
            if str(path) == "/proc/vmstat":
                return "pswpin 0\npswpout 0\n"
            return read_text(path, *args, **kwargs)

        def fake_probe(command, timeout=10):
            if any("--query-compute-apps" in part for part in command):
                return result()
            if any("--query-gpu=uuid" in part for part in command):
                return result(status="unavailable") if missing_gpu else result(
                    "GPU-test,0,N/A,N/A,40,8,300,P8\n")
            return result()

        def fake_sleep(duration):
            self.clock += duration

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(Path, "read_text", fake_read))
            stack.enter_context(mock.patch.object(inventory, "probe", side_effect=fake_probe))
            stack.enter_context(mock.patch.object(inventory.time, "sleep", side_effect=fake_sleep))
            stack.enter_context(mock.patch.object(inventory.time, "monotonic", side_effect=lambda: self.clock))
            stack.enter_context(mock.patch.object(inventory.socket, "gethostname", return_value=host))
            stack.enter_context(mock.patch.object(manifest, "capture_git", return_value={
                "commit": "a" * 40, "dirty": False, "root": str(self.root),
                "branch": "main", "changed_paths": [], "error": None}))
            stack.enter_context(mock.patch.object(manifest, "capture_system", return_value={
                "hostname": host, "machine": "aarch64"}))
            stack.enter_context(mock.patch.object(manifest, "capture_packages", return_value={}))
            stack.enter_context(mock.patch.object(manifest, "capture_nvidia", return_value={}))
            stack.enter_context(mock.patch.object(inventory.subprocess, "run", side_effect=AssertionError("no subprocess allowed")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return inventory.capture(self.config_path, self.root / "results")

    def test_capture_preserves_sources_raw_manifest_and_timing(self):
        directory = self.capture()
        document = manifest.load_run(directory)
        assessment = json.loads((directory / "raw/assessment.json").read_text())
        self.assertEqual(document["status"], "complete")
        self.assertIsNone(document["model"])
        self.assertTrue((directory / "COMPLETE").exists())
        self.assertEqual(assessment["idle_status"], "validated")
        self.assertEqual(assessment["sample_count"], 2)
        self.assertEqual(assessment["elapsed_seconds"], 60)
        self.assertEqual(len((directory / "metrics/idle.jsonl").read_text().splitlines()), 2)
        self.assertTrue((directory / "raw/sources/inventory.py").is_file())
        self.assertTrue((directory / "raw/sources/run-context.schema.json").is_file())
        self.assertEqual(document["schema_version"], manifest.SCHEMA_VERSION)
        self.assertEqual(document["hosts"], ["test-pegasus"])
        self.assertTrue((directory / "raw/context.json").is_file())
        self.assertTrue(all(record["matches"] for record in document["input_verification"]))

    def test_wrong_host_is_rejected_before_run_directory(self):
        with self.assertRaisesRegex(ValueError, "host inesperado"):
            self.capture(host="octopus")
        self.assertFalse((self.root / "results").exists())

    def test_capture_exception_marks_failed_and_preserves_evidence(self):
        with self.assertRaisesRegex(OSError, "synthetic proc failure"):
            self.capture(broken_cpu=True)
        directory, = (self.root / "results/system-baseline").iterdir()
        document = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(document["status"], "failed")
        self.assertTrue(document["errors"])
        self.assertTrue((directory / "FAILED").exists())
        self.assertTrue((directory / "raw/inventory.json").exists())

    def test_capture_missing_required_gpu_evidence_cannot_complete(self):
        try:
            self.capture(missing_gpu=True)
        except manifest.ManifestError:
            pass
        directory, = (self.root / "results/system-baseline").iterdir()
        document = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(document["status"], "failed")
        self.assertTrue(document["errors"])
        self.assertTrue((directory / "FAILED").exists())
        self.assertFalse((directory / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
