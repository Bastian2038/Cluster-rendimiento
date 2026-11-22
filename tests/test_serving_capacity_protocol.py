"""Check the public preparation commands and the fixed four/two/one design on CPU."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from infraestructura import serving_capacity as cli
from infraestructura import serving_capacity_protocol as protocol


class CapacityPreparationTests(unittest.TestCase):
    def test_plan_matches_order_commands_counts_and_budget(self):
        plan = protocol.make_plan()
        self.assertEqual(plan["counts"], {
            "jobs": 18, "blocks": 3, "points": 126,
            "warmup_requests": 2016, "measured_requests": 16128,
        })
        expected = [
            (block, model, group, nodes)
            for block in (1, 2, 3)
            for model in ("qwen35-9b", "llama31-8b")
            for group, nodes in (
                ("four", ["pegasus", "octopus", "griffin", "draggon"]),
                ("pair", ["pegasus", "octopus"]), ("single", ["pegasus"]),
            )
        ]
        actual = [(j["block"], j["model"]["key"], j["group"], j["nodes"]) for j in plan["jobs"]]
        self.assertEqual(actual, expected)
        for job in plan["jobs"]:
            self.assertEqual(tuple(cell["rate_rps"] for cell in job["cells"]), protocol.RATES)
            self.assertEqual(set(job["ray_commands"]), set(job["nodes"]))
            for cell in job["cells"]:
                for phase, count in (("warmup", 16), ("measured", 128)):
                    cmd = cell["commands"][phase]
                    self.assertEqual(cmd[cmd.index("--count") + 1], str(count))
                    self.assertEqual(cmd[cmd.index("--rate") + 1], str(cell["rate_rps"]))
                    self.assertEqual(cmd[cmd.index("--phase") + 1], phase)
        self.assertEqual(plan["budget"]["measured_emission_seconds"], 23731.2)
        self.assertEqual(plan["budget"]["warmup_emission_seconds"], 2966.4)
        self.assertEqual(plan["budget"]["emission_seconds_per_block"], 8899.2)
        self.assertEqual(plan["budget"]["proposed_reservation_seconds"], 43200)

    def test_modified_design_and_reference_are_rejected(self):
        config = json.loads(protocol.DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for group, field, value in (
                ("capacity", "rates_rps", [0.25, 2]),
                ("capacity", "compliance", float("nan")),
                ("measurement", "blocks", 2),
                ("runtime", "prefix_caching", True),
                ("paths", "workspace", "/tmp/different"),
            ):
                with self.subTest(field=field):
                    changed = deepcopy(config)
                    changed[group][field] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        protocol.load_config(path)
            changed = deepcopy(config)
            changed["models"][0]["tokenizer_revision"] = "different"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "models"):
                protocol.load_config(path)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                protocol.validate_reference_index(config, path)

    def test_public_preparation_commands_have_no_live_or_campaign_side_effects(self):
        with (tempfile.TemporaryDirectory() as tmp,
              mock.patch.object(cli.inference_runtime, "preflight") as live,
              mock.patch.object(cli.inference_runtime, "remote") as remote,
              mock.patch.object(protocol.manifest, "start_run") as start,
              mock.patch.object(protocol, "frozen_workloads", return_value=Path(tmp)) as frozen,
              mock.patch.object(protocol, "verify_hardware", return_value={}) as hardware,
              redirect_stderr(io.StringIO())):
            output = Path(tmp) / "plan.json"
            self.assertEqual(cli.main(["plan", "--output", str(output)]), 0)
            original = output.read_bytes()
            self.assertEqual(cli.main(["plan", "--output", str(output)]), 2)
            self.assertEqual(output.read_bytes(), original)
            preflight = Path(tmp) / "preflight.json"
            self.assertEqual(cli.main(["preflight", "--output", str(preflight)]), 0)
            result = json.loads(preflight.read_text())
            self.assertEqual((result["scope"], result["errors"]), ("local_cpu", []))
            self.assertEqual(len(result["historical_jsonl_sha256"]), 8)
            frozen.assert_called_once()
            hardware.assert_called_once()
            self.assertEqual(cli.main(["run"]), 2)
            for protected in ("resultados/raw", "datos"):
                with self.assertRaises(ValueError):
                    cli._check_destination(protocol.REPO_ROOT / protected / "plan.json")
            live.assert_not_called()
            remote.assert_not_called()
            start.assert_not_called()

    def test_rejects_replacement_gpu_driver_and_environment_even_with_a_matching_new_roster(self):
        from tests.test_inference import ready_probe

        config, inventory, _ = protocol.load_config()
        history = {"status": "ready_for_smoke", "errors": [], "nodes": [
            ready_probe(config, inventory, node) for node in inventory["nodes"]]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.json"
            protocol.inference.write_new(path, history)
            digest = protocol.manifest.sha256_file(path)
            config["reference"]["historical_preflight_sha256"] = digest
            with mock.patch.object(protocol, "HISTORICAL_PREFLIGHT_SHA256", digest):
                self.assertEqual(len(protocol.verify_hardware(config, path, history)), 4)
                for field in ("gpu_uuid", "driver", "hostname", "entorno"):
                    with self.subTest(field=field):
                        observed = deepcopy(history)
                        node = next(n for n in observed["nodes"] if n["node_id"] == "griffin")
                        if field in ("gpu_uuid", "driver"):
                            gpu = node["gpu"]["stdout"].strip().split(", ")
                            gpu[0 if field == "gpu_uuid" else 2] = "GPU-reemplazo" if field == "gpu_uuid" else "nuevo"
                            node["gpu"]["stdout"] = ", ".join(gpu)
                        elif field == "hostname":
                            node["hostname"] = "otro-host"
                        else:
                            node["image"]["stdout"] = "otra-imagen"
                        with self.assertRaisesRegex(ValueError, f"griffin.*{field}"):
                            protocol.verify_hardware(config, path, observed)
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    protocol.verify_hardware(config, path, history)

    def test_operator_receives_the_directory_of_a_closed_partial_campaign(self):
        from infraestructura import serving_capacity_runner as runner

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            directory = Path(tmp)
            (directory / "campaign.json").write_text('{"status":"partial"}')
            with mock.patch.object(runner, "run_campaign", return_value=directory) as run:
                self.assertEqual(cli.main(["run", "--execute", "--session-limit-seconds", "3600"]), 2)
            run.assert_called_once_with(protocol.DEFAULT_CONFIG, session_limit_seconds=3600)
            self.assertEqual(json.loads(output.getvalue()), {
                "event": "campaign_closed", "directory": str(directory), "status": "partial",
            })
