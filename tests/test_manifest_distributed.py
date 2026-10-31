"""Contratos de evidencia distribuida, cierre inmutable y lectura histórica."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infraestructura import manifest


class DistributedManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def start(self, kind="cluster-inventory"):
        with mock.patch.object(manifest, "capture_git", return_value={
            "commit": "b" * 40, "dirty": False, "root": str(self.root),
            "branch": "main", "changed_paths": [], "error": None,
        }), mock.patch.object(manifest, "capture_nvidia", return_value={
            "nvidia_smi_available": False, "gpus": [], "error": None,
        }), mock.patch.object(manifest, "capture_packages", return_value={}):
            return manifest.start_run(
                kind=kind, model_id=None, model_revision=None, tokenizer_id=None,
                tokenizer_revision=None, requested_config={"hosts": ["inventado"]},
                results_root=self.root, repo_path=self.root,
            )

    def evidence(self, run, name="pegasus", rank=None, gpu="GPU-1"):
        identity = {"node_id": name, "hostname": "host-" + name,
                    "gpu_uuid": gpu, "rank": rank}
        relative = "raw/" + name + ".json"
        (run / relative).write_text(json.dumps(identity))
        return {**identity, "evidence_path": relative}

    def test_requested_hosts_never_become_participants(self):
        run = self.start()
        self.assertEqual(manifest.load_run(run)["hosts"], [])
        with self.assertRaisesRegex(manifest.ManifestError, "sin participantes"):
            manifest.finalize_run(run)
        manifest.finalize_run(run, status="failed", errors=["SSH falló"])
        self.assertEqual(manifest.load_run(run)["hosts"], [])
        manifest.verify_artifacts(run)

    def test_register_verify_and_preserve_closed_bytes(self):
        run = self.start()
        manifest.register_participants(run, [self.evidence(run)])
        self.assertEqual(manifest.load_run(run)["hosts"], ["host-pegasus"])
        manifest.finalize_run(run)
        before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
        manifest.verify_artifacts(run)
        with self.assertRaisesRegex(manifest.ManifestError, "cerrada"):
            manifest.register_participants(run, [])
        with self.assertRaisesRegex(manifest.ManifestError, "finalizada"):
            manifest.finalize_run(run)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_duplicate_identity_rank_gpu_rejected_atomically(self):
        for field in ("node_id", "hostname", "rank", "gpu_uuid"):
            with self.subTest(field=field):
                run = self.start("nccl-test")
                first = self.evidence(run, rank=0)
                second = self.evidence(run, "octopus", rank=1, gpu="GPU-2")
                second[field] = first[field]
                (run / second["evidence_path"]).write_text(json.dumps(second))
                with self.assertRaisesRegex(manifest.ManifestError, "duplicado"):
                    manifest.register_participants(run, [first, second])
                self.assertEqual(manifest.load_run(run)["participants"], [])

    def test_missing_rank_or_gpu_is_not_nccl_participation(self):
        run = self.start("nccl-test")
        for rank, gpu in ((None, "GPU-1"), (0, None)):
            with self.assertRaisesRegex(manifest.ManifestError, "requiere GPU y rank"):
                manifest.register_participants(run, [self.evidence(run, rank=rank, gpu=gpu)])

    def test_claimed_identity_must_match_source(self):
        run = self.start()
        evidence = self.evidence(run)
        evidence["hostname"] = "otro"
        with self.assertRaisesRegex(manifest.ManifestError, "identidad"):
            manifest.register_participants(run, [evidence])

    def test_tampered_participant_source_rejected_even_after_close(self):
        run = self.start()
        manifest.register_participants(run, [self.evidence(run)])
        manifest.finalize_run(run)
        with (run / "raw/pegasus.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(manifest.ManifestError, "SHA-256"):
            manifest.load_run(run)

    def test_evidence_escape_and_external_symlink_rejected(self):
        run = self.start()
        evidence = self.evidence(run)
        outside = self.root / "outside.json"
        outside.write_bytes((run / evidence["evidence_path"]).read_bytes())
        for relative in (str(outside), "../outside.json"):
            with self.assertRaisesRegex(manifest.ManifestError, "fuera"):
                manifest.register_participants(run, [{**evidence, "evidence_path": relative}])
        (run / "raw/link.json").symlink_to(outside)
        with self.assertRaisesRegex(manifest.ManifestError, "fuera"):
            manifest.register_participants(run, [{**evidence, "evidence_path": "raw/link.json"}])

    def test_report_integrity_detects_changed_and_added_files(self):
        run = self.start()
        manifest.register_participants(run, [self.evidence(run)])
        log = run / "logs/stdout.txt"
        log.write_text("original")
        manifest.finalize_run(run)
        log.write_text("alterado")
        with self.assertRaisesRegex(manifest.ManifestError, "integridad"):
            manifest.verify_artifacts(run)
        log.write_text("original")
        (run / "raw/added.json").write_text("{}")
        with self.assertRaisesRegex(manifest.ManifestError, "índice"):
            manifest.verify_artifacts(run)

    def test_historical_20_read_does_not_rewrite_or_upgrade(self):
        run = self.start("system-baseline")
        # Construye fixture histórico antes del cierre, nunca toca evidencia real.
        manifest_path, context_path = run / "manifest.json", run / "raw/context.json"
        core = json.loads(manifest_path.read_text())
        context = json.loads(context_path.read_text())
        core["schema_version"] = context["schema_version"] = "2.0"
        context_path.write_text(json.dumps(context))
        core["context"]["sha256"] = manifest.sha256_file(context_path)
        manifest_path.write_text(json.dumps(core))
        manifest.finalize_run(run)
        before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
        self.assertEqual(manifest.load_run(run)["schema_version"], "2.0")
        manifest.verify_artifacts(run)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_new_contract_validates_schemas_and_legacy_rejects_participants(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema no instalado")
        run = self.start("nccl-test")
        manifest.register_participants(run, [self.evidence(run, rank=0)])
        for document, schema_name in (("manifest.json", "run-manifest.schema.json"),
                                      ("raw/context.json", "run-context.schema.json")):
            data = json.loads((run / document).read_text())
            schema = json.loads((Path(manifest.__file__).parent / "schemas" / schema_name).read_text())
            validator = jsonschema.Draft202012Validator(schema)
            validator.validate(data)
            self.assertFalse(validator.is_valid({**data, "schema_version": "2.0"}))


if __name__ == "__main__":
    unittest.main()
