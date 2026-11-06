"""Contrato de manifiesto para una corrida de inferencia distribuida."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infraestructura import manifest


class InferenceManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def start(self, **overrides):
        params = dict(kind="inference-serving", model_id="Qwen/model",
                      model_revision="a" * 40, tokenizer_id="Qwen/model",
                      tokenizer_revision="b" * 40,
                      requested_config={"hosts": ["host-inventado"]},
                      results_root=self.root, repo_path=self.root)
        params.update(overrides)
        with mock.patch.object(manifest, "capture_git", return_value={
            "commit": "c" * 40, "dirty": False, "root": str(self.root),
            "branch": "main", "changed_paths": [], "error": None,
        }), mock.patch.object(manifest, "capture_nvidia", return_value={
            "nvidia_smi_available": False, "gpus": [], "error": None,
        }), mock.patch.object(manifest, "capture_packages", return_value={}):
            return manifest.start_run(**params)

    def evidence(self, run, rank=0, gpu="GPU-1"):
        item = {"node_id": "pegasus", "hostname": "host-pegasus",
                "gpu_uuid": gpu, "rank": rank}
        (run / "raw/pegasus.json").write_text(json.dumps(item))
        return {**item, "evidence_path": "raw/pegasus.json"}

    def test_requested_host_does_not_complete_without_observed_participants(self):
        run = self.start()
        self.assertEqual(manifest.load_run(run)["hosts"], [])
        with self.assertRaisesRegex(manifest.ManifestError, "sin participantes"):
            manifest.finalize_run(run)
        self.assertFalse((run / "COMPLETE").exists())
        manifest.register_participants(run, [self.evidence(run)])
        manifest.finalize_run(run)
        self.assertEqual(manifest.load_run(run)["hosts"], ["host-pegasus"])

    def test_requires_model_tokenizer_and_pinned_revisions(self):
        for change in ({"model_id": None}, {"tokenizer_id": None},
                       {"model_revision": None}, {"tokenizer_revision": "main"},
                       {"model_revision": "main", "allow_floating_revision": True}):
            with self.subTest(change=change), self.assertRaises(manifest.ManifestError):
                self.start(**change)

    def test_gpu_and_rank_must_be_observed(self):
        run = self.start()
        for rank, gpu in ((None, "GPU-1"), (0, None)):
            with self.subTest(rank=rank, gpu=gpu):
                with self.assertRaisesRegex(manifest.ManifestError, "requiere GPU y rank"):
                    manifest.register_participants(run, [self.evidence(run, rank, gpu)])
                self.assertEqual(manifest.load_run(run)["participants"], [])

    def test_load_rejects_tampered_null_model(self):
        run = self.start()
        compact = run / "manifest.json"
        data = json.loads(compact.read_text())
        data["model"] = None
        compact.write_text(json.dumps(data))
        with self.assertRaisesRegex(manifest.ManifestError, "modelo/tokenizer"):
            manifest.load_run(run)

    def test_schema_rejects_null_model_and_unpinned_revision(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema no instalado")
        run = self.start()
        data = json.loads((run / "manifest.json").read_text())
        schema = json.loads((Path(manifest.__file__).parent / "schemas/run-manifest.schema.json").read_text())
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(data)
        self.assertFalse(validator.is_valid({**data, "model": None}))
        self.assertFalse(validator.is_valid({**data, "model": {**data["model"], "revision": "main"}}))


if __name__ == "__main__":
    unittest.main()
