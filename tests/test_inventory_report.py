import json
import tempfile
import unittest
from pathlib import Path

from infraestructura import inventory, inventory_report, manifest
from tests.test_inventory import config, result, sample


class InventoryReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "raw" / "fixture"
        (self.run / "raw").mkdir(parents=True)
        (self.run / "metrics").mkdir()
        samples = [sample(), sample()]
        observed = {"probes": {name: {**result(), "stderr": ""} for name in inventory.COMMANDS}}
        assessment = inventory.summarize(samples, config(), observed)
        assessment.update(elapsed_seconds=60, swap_io_pages={"pswpin": 0, "pswpout": 0})
        inventory.write_json(self.run / "raw/assessment.json", assessment)
        inventory.write_json(self.run / "raw/inventory.json", observed)
        (self.run / "metrics/idle.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
        self.document = {
            "schema_version": "2.0", "kind": "system-baseline", "status": "complete", "run_id": "fixture",
            "started_at": "2026-09-16T00:00:00Z", "ended_at": "2026-09-16T00:01:00Z",
            "system": {"hostname": "test", "machine": "aarch64"},
            "git": {"commit": "a" * 40, "dirty": True},
            "policy": {"strict": False, "exceptions": ["dirty_worktree"]},
            "model": None, "errors": [], "accelerator": {},
            "requested_config": config(), "requested_config_file": None,
            "inputs": [], "input_verification": [],
            "artifacts": manifest._artifact_records(self.run)}
        manifest._write_compact_run(self.run, self.document)
        (self.run / "COMPLETE").write_text(self.document["ended_at"] + "\n")
        self.output = self.root / "report.md"

    def test_render_links_integrity_and_marks_missing_metrics(self):
        before = {path.relative_to(self.run): path.read_bytes()
                  for path in self.run.rglob("*") if path.is_file()}
        inventory_report.render(self.run, self.output)
        content = self.output.read_text()
        self.assertIn("validated", content)
        self.assertIn("no disponible", content)
        self.assertIn(manifest.sha256_file(self.run / "manifest.json"), content)
        after = {path.relative_to(self.run): path.read_bytes()
                 for path in self.run.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_rejects_output_inside_run(self):
        with self.assertRaisesRegex(ValueError, "fuera de raw"):
            inventory_report.render(self.run, self.run / "report.md")

    def test_rejects_overwriting_report(self):
        inventory_report.render(self.run, self.output)
        with self.assertRaises(FileExistsError):
            inventory_report.render(self.run, self.output)

    def test_rejects_corrupt_artifact(self):
        (self.run / "metrics/idle.jsonl").write_text("corrupt test fixture")
        with self.assertRaisesRegex(ValueError, "integridad"):
            inventory_report.render(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_running_manifest(self):
        self.document["status"] = "running"
        self.document["ended_at"] = None
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "finalizado"):
            inventory_report.render(self.run, self.output)

    def refresh_manifest(self):
        self.document["artifacts"] = manifest._artifact_records(self.run)
        manifest._write_compact_run(self.run, self.document)
        for name in ("COMPLETE", "FAILED"):
            (self.run / name).unlink(missing_ok=True)
        if self.document["status"] in {"complete", "failed"}:
            marker = "COMPLETE" if self.document["status"] == "complete" else "FAILED"
            (self.run / marker).write_text(self.document["ended_at"] + "\n")

    def test_compact_report_has_no_command_dumps_and_retains_cpu_models(self):
        path = self.run / "raw/inventory.json"
        observed = json.loads(path.read_text())
        observed["probes"]["cpu"] = result(json.dumps({"lscpu": [
            {"field": "CPU(s):", "data": "20"},
            {"field": "Model name:", "data": "Cortex-X925"},
            {"field": "Model name:", "data": "Cortex-A725"}]}))
        observed["probes"]["gpu"] = result("NVIDIA GB10, GPU-fixture, 580, [N/A]\n")
        observed["probes"]["processes"] = result("DO_NOT_DUMP_PROCESS_LIST")
        path.write_text(json.dumps(observed))
        self.refresh_manifest()
        inventory_report.render(self.run, self.output)
        content = self.output.read_text()
        self.assertIn("Cortex-X925 + Cortex-A725 (20 CPU lógicas)", content)
        self.assertIn("NVIDIA GB10", content)
        self.assertNotIn("DO_NOT_DUMP_PROCESS_LIST", content)
        self.assertNotIn("```", content)
        self.assertIn("[Manifiesto](raw/fixture/manifest.json)", content)
        self.assertLess(len(content.splitlines()), 80)

    def test_rejects_legacy_versions_without_creating_report(self):
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                historical = {**self.document, "schema_version": version}
                (self.run / "manifest.json").write_text(json.dumps(historical))
                output = self.root / f"report-{version}.md"
                with self.assertRaisesRegex(manifest.ManifestError, "schema_version no soportada"):
                    inventory_report.render(self.run, output)
                self.assertFalse(output.exists())

    def test_renders_compact_and_verifies_context(self):
        inventory_report.render(self.run, self.output)
        self.assertIn("[Contexto técnico](raw/fixture/raw/context.json)", self.output.read_text())
        context = self.run / "raw/context.json"
        context.write_text(context.read_text() + " ")
        with self.assertRaisesRegex(manifest.ManifestError, "contexto"):
            inventory_report.render(self.run, self.root / "corrupt-context.md")

    def test_rejects_missing_context(self):
        (self.run / "raw/context.json").unlink()
        with self.assertRaisesRegex(manifest.ManifestError, "contexto"):
            inventory_report.render(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_missing_registered_artifact_even_when_failed(self):
        self.document["status"] = "failed"
        self.refresh_manifest()
        (self.run / "raw/assessment.json").unlink()
        with self.assertRaisesRegex(ValueError, "integridad"):
            inventory_report.render(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_renders_partial_failed_capture_without_inventing_measurements(self):
        self.document["status"] = "failed"
        self.document["errors"] = [{"message": "KeyboardInterrupt"}]
        for relative in ("raw/assessment.json", "raw/inventory.json", "metrics/idle.jsonl"):
            (self.run / relative).unlink()
        self.refresh_manifest()
        inventory_report.render(self.run, self.output)
        content = self.output.read_text()
        self.assertIn("**failed**", content)
        self.assertIn("**unverified**", content)
        self.assertIn("KeyboardInterrupt", content)
        self.assertIn("Evidencia ausente: `raw/assessment.json`", content)
        self.assertIn("no disponible/no disponible", content)
        self.assertNotIn("0.000", content)

    def test_marks_initial_and_final_probe_failures_and_threshold_violations(self):
        path = self.run / "raw/inventory.json"
        observed = json.loads(path.read_text())
        observed["probes"]["cuda_compiler"] = result(status="unavailable")
        path.write_text(json.dumps(observed))
        (self.run / "raw/after.json").write_text(json.dumps({"containers": result(status="error")}))
        path = self.run / "raw/assessment.json"
        assessment = json.loads(path.read_text())
        assessment.update(idle_status="not_idle", violations=["cpu_busy_p95"])
        path.write_text(json.dumps(assessment))
        self.refresh_manifest()
        inventory_report.render(self.run, self.output)
        content = self.output.read_text()
        self.assertIn("Captura: **complete**; criterios de reposo: **not_idle**", content)
        self.assertIn("cuda_compiler (unavailable)", content)
        self.assertIn("containers (error)", content)
        self.assertIn("Criterio incumplido: `cpu_busy_p95`", content)
        self.assertIn("dirty_worktree", content)

    def test_rejects_evidence_added_after_finalization(self):
        (self.run / "raw/after.json").write_text(json.dumps({"containers": result()}))
        with self.assertRaisesRegex(ValueError, "no registrada"):
            inventory_report.render(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_artifact_records(self):
        self.document["artifacts"].append(self.document["artifacts"][0])
        manifest._write_compact_run(self.run, self.document)
        with self.assertRaisesRegex(ValueError, "duplicado"):
            inventory_report.render(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_manifest_without_completion_marker(self):
        (self.run / "COMPLETE").unlink()
        with self.assertRaisesRegex(manifest.ManifestError, "cierre incompleto"):
            inventory_report.render(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_reports_gpu_process_presence_even_with_partial_visibility(self):
        samples = [sample(), sample()]
        samples[0]["gpu_processes"] = [{"pid": 123}]
        samples[1]["gpu_processes"] = None
        (self.run / "metrics/idle.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
        self.refresh_manifest()
        inventory_report.render(self.run, self.output)
        self.assertIn("presentes (cobertura incompleta)", self.output.read_text())
