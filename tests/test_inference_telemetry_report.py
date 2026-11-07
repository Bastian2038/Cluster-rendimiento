"""Synthetic CPU-only identity checks for the telemetry analyzer."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infraestructura import inference_telemetry_report as module


class AnalysisIdentityTests(unittest.TestCase):
    def test_window_units_and_counter_rates(self):
        def sample(second, memory_kib, cpu, first_rx, second_rx):
            return {
                "_utc_seconds": second,
                "timestamp_utc": f"2026-01-01T00:00:{second + 10:02d}+00:00",
                "monotonic_ns": (second + 2) * 1_000_000_000,
                "memory": {"MemAvailable_kib": memory_kib},
                "cpu_ticks": dict(zip(("user", "system", "idle", "iowait"), cpu)),
                "gpus": [],
                "nic_f0": {
                    "enp1s0f0np0": {"rx_bytes": first_rx, "tx_bytes": first_rx * 2},
                    "enP2p1s0f0np0": {"rx_bytes": second_rx, "tx_bytes": second_rx * 3},
                },
            }

        samples = [
            sample(-1, 99 * 1_048_576, (0, 0, 0, 0), 0, 0),
            sample(0, 1_048_576, (10, 10, 80, 0), 100, 20),
            sample(1, 2_097_152, (20, 20, 150, 10), 200, 50),
            sample(2, 3_145_728, (50, 30, 200, 20), 500, 90),
            sample(3, 99 * 1_048_576, (999, 999, 999, 999), 9999, 9999),
        ]
        result = module.summarize(samples, 0, 2)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["mem_available_gib"]["median"], 2)
        self.assertEqual(result["mem_available_gib"]["max"], 3)
        self.assertEqual(result["cpu_busy"]["min"], 20)
        self.assertEqual(result["cpu_busy"]["median"], 30)
        self.assertEqual(result["cpu_busy"]["max"], 40)
        first = result["nic_f0"]["enp1s0f0np0"]
        second = result["nic_f0"]["enP2p1s0f0np0"]
        self.assertEqual(first["rx_bytes_per_second"]["median"], 200)
        self.assertEqual(first["tx_bytes_per_second"]["median"], 400)
        self.assertEqual(second["rx_bytes_per_second"]["median"], 35)
        self.assertEqual(second["tx_bytes_per_second"]["median"], 105)

    def test_nonfinite_values_are_null(self):
        stat = module.describe([float("nan"), float("inf"), 5], "W")
        self.assertEqual(stat["valid_samples"], 1)
        self.assertEqual(stat["null_samples"], 2)
        self.assertEqual(stat["median"], 5)

    def test_rejects_forged_labels_and_t32(self):
        with tempfile.TemporaryDirectory(prefix="ipre-telemetry-fixture-") as temp:
            root = Path(temp)
            run_dir = root / "run"
            (run_dir / "metrics").mkdir(parents=True)
            telemetry = run_dir / "raw/nodes/pegasus/logs/telemetry.jsonl"
            telemetry.parent.mkdir(parents=True)
            (run_dir / "COMPLETE").write_text("closed")
            (run_dir / "manifest.json").write_text("{}")
            cell = {"cell": "512-c1", "status": "valid", "input_tokens": 512,
                    "concurrency": 1, "t32_seconds": 1.0,
                    "measurement_started_at": "2026-01-01T00:00:00+00:00",
                    "measurement_ended_at": "2026-01-01T00:00:02+00:00",
                    "model_key": "qwen", "group": "four", "block": 1,
                    "run_dir": str(run_dir)}
            (run_dir / "metrics/assessment.json").write_text(
                json.dumps({"valid": True, "cells": [cell]}))
            telemetry.write_text("\n".join(json.dumps({
                "timestamp_utc": f"2026-01-01T00:00:0{n}+00:00",
                "monotonic_ns": n * 1_000_000_000, "hostname": "pegasus",
                "memory": {"MemAvailable_kib": 1_048_576}, "gpus": [],
            }) for n in (0, 1, 2)) + "\n")
            report = {"campaign_id": "campaign", "runs": [{"run_dir": str(run_dir),
                      "manifest_sha256": module.digest(run_dir / "manifest.json"),
                      "model_key": "qwen", "group": "four", "block": 1}],
                      "cells": [cell.copy()]}
            report_path = root / "report.json"
            run = {"status": "complete", "kind": "inference-serving",
                   "requested_config": {"campaign_id": "campaign",
                                        "job": {"name": "job", "group": "four", "block": 1,
                                                "model": {"key": "qwen"}}},
                   "participants": [{"node_id": "pegasus", "hostname": "pegasus"}]}
            with patch.object(module.manifest, "load_run", return_value=run), \
                 patch.object(module.manifest, "verify_artifacts"):
                report_path.write_text(json.dumps(report))
                self.assertEqual(len(module.analyze(report_path)["rows"]), 1)
                report["cells"][0]["model_key"] = "llama"
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "identity"):
                    module.analyze(report_path)
                report["cells"][0]["model_key"] = "qwen"
                report["cells"][0]["t32_seconds"] = 9.0
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "assessment"):
                    module.analyze(report_path)


if __name__ == "__main__":
    unittest.main()
