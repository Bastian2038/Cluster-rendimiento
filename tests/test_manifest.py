import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infraestructura import manifest

try:
    import jsonschema
except ImportError:
    jsonschema = None


PINNED_REVISION = "a" * 40


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "dataset.csv"
        self.input_path.write_text("item,value\n1,ok\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _start(self, **overrides):
        arguments = {
            "kind": "sesgo",
            "model_id": "organization/model",
            "model_revision": PINNED_REVISION,
            "tokenizer_id": "organization/model",
            "tokenizer_revision": PINNED_REVISION,
            "requested_config": {"temperature": 0},
            "inputs": {"dataset": self.input_path},
            "results_root": self.root / "resultados" / "raw",
            "run_id": "test-run",
            "repo_path": self.root,
        }
        arguments.update(overrides)
        with (
            mock.patch.object(
                manifest,
                "capture_git",
                return_value={
                    "commit": "b" * 40,
                    "dirty": False,
                    "root": str(self.root),
                    "branch": "main",
                    "changed_paths": [],
                    "error": None,
                },
            ),
            mock.patch.object(
                manifest,
                "capture_nvidia",
                return_value={
                    "nvidia_smi_available": False,
                    "gpus": [],
                    "error": None,
                },
            ),
            mock.patch.object(manifest, "capture_packages", return_value={}),
        ):
            return manifest.start_run(**arguments)

    def test_run_command_preserves_porcelain_leading_space(self):
        completed = mock.Mock(stdout=" M .gitignore\n")
        with mock.patch.object(manifest.subprocess, "run", return_value=completed):
            self.assertEqual(manifest._run_command(["git", "status"]), " M .gitignore")

    def test_start_writes_manifest_with_input_hash(self):
        run_directory = self._start()
        content = manifest.load_run(run_directory)

        self.assertEqual(content["status"], "running")
        self.assertEqual(content["schema_version"], "2.1")
        self.assertEqual(content["git"]["commit"], "b" * 40)
        self.assertEqual(content["model"]["revision"], PINNED_REVISION)
        self.assertEqual(content["requested_config"], {"temperature": 0})
        self.assertEqual(content["inputs"][0]["size_bytes"], 16)
        self.assertEqual(
            content["inputs"][0]["sha256"], manifest.sha256_file(self.input_path)
        )
        self.assertEqual(content["git"]["branch"], "main")
        self.assertTrue((run_directory / "raw").is_dir())
        self.assertTrue((run_directory / "logs").is_dir())
        self.assertTrue((run_directory / "metrics").is_dir())
        self.assertFalse(list(run_directory.glob("*.tmp")))

    def _start_system(self, **overrides):
        arguments = dict(
            kind="system-baseline",
            model_id=None,
            model_revision=None,
            tokenizer_id=None,
            tokenizer_revision=None,
        )
        arguments.update(overrides)
        return self._start(**arguments)

    def test_system_baseline_has_null_model_without_revision_exception(self):
        directory = self._start_system()
        content = manifest.load_run(directory)
        self.assertEqual(content["schema_version"], "2.1")
        self.assertIsNone(content["model"])
        self.assertEqual(content["policy"], {"strict": True, "exceptions": []})
        manifest.finalize_run(directory)
        self.assertTrue((directory / "COMPLETE").exists())

    def test_system_baseline_dirty_exception_is_explicit(self):
        directory = self._start_system(allow_dirty=True)
        content = manifest.load_run(directory)
        self.assertEqual(content["policy"], {
            "strict": False, "exceptions": ["dirty_worktree"]
        })

    def test_system_baseline_rejects_model_metadata_and_floating_exception(self):
        for field in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(manifest.ManifestError, "null"):
                    self._start_system(**{field: "not-applicable"})
        with self.assertRaisesRegex(manifest.ManifestError, "no aplica"):
            self._start_system(allow_floating_revision=True)

    def test_null_model_metadata_rejected_for_model_runs(self):
        for kind in ("sesgo", "rendimiento", "sft", "smoke"):
            for field in ("model_id", "tokenizer_id"):
                with self.subTest(kind=kind, field=field):
                    with self.assertRaisesRegex(manifest.ManifestError, "obligatorios"):
                        self._start(kind=kind, allow_floating_revision=True, **{field: None})

    def test_system_baseline_preserves_git_guards(self):
        for git, message in (
            ({"commit": None, "dirty": None}, "commit Git"),
            ({"commit": "b" * 40, "dirty": True}, "dirty"),
        ):
            with self.subTest(git=git):
                with mock.patch.object(manifest, "capture_git", return_value=git):
                    with self.assertRaisesRegex(manifest.ManifestError, message):
                        manifest.start_run(
                            kind="system-baseline", model_id=None, model_revision=None,
                            tokenizer_id=None, tokenizer_revision=None,
                            requested_config={}, results_root=self.root / "raw",
                        )

    def test_system_baseline_cli_accepts_omitted_model_arguments(self):
        args = manifest.build_parser().parse_args([
            "start", "--kind", "system-baseline", "--config", "config.json"
        ])
        self.assertIsNone(args.model_id)
        self.assertIsNone(args.model_revision)
        self.assertIsNone(args.tokenizer_id)
        self.assertIsNone(args.tokenizer_revision)

    @unittest.skipIf(jsonschema is None, "jsonschema no está instalado")
    def test_schemas_validate_only_compact_contracts(self):
        schemas = Path(manifest.__file__).parent / "schemas"
        schema = json.loads((schemas / "run-manifest.schema.json").read_text())
        context_schema = json.loads((schemas / "run-context.schema.json").read_text())
        for document in (schema, context_schema):
            jsonschema.Draft202012Validator.check_schema(document)
        validator = jsonschema.Draft202012Validator(schema)
        context_validator = jsonschema.Draft202012Validator(context_schema)
        for directory in (self._start(), self._start_system()):
            compact = json.loads((directory / "manifest.json").read_text())
            context = json.loads((directory / "raw/context.json").read_text())
            validator.validate(compact)
            context_validator.validate(context)
            legacy = manifest.load_run(directory)
            legacy.pop("hosts")
            legacy.pop("context")
            for version in ("1.0", "1.1"):
                with self.subTest(kind=compact["kind"], version=version):
                    legacy["schema_version"] = version
                    self.assertFalse(validator.is_valid(legacy))
                    self.assertFalse(validator.is_valid({**compact, "schema_version": version}))
                    self.assertFalse(context_validator.is_valid({**context, "schema_version": version}))
            self.assertFalse(validator.is_valid({**compact, "policy": {}}))
            self.assertFalse(context_validator.is_valid({**context, "status": "complete"}))
            bad_model = {"id": "x", "revision": "a" * 40,
                         "tokenizer": {"id": "x", "revision": "a" * 40}}
            bad_model = bad_model if compact["model"] is None else None
            self.assertFalse(validator.is_valid({**compact, "model": bad_model}))

    def test_start_rejects_run_id_collision(self):
        self._start()
        with self.assertRaisesRegex(manifest.ManifestError, "ya existe"):
            self._start()

    def test_requested_config_file_is_hashed(self):
        config_path = self.root / "config.json"
        config_path.write_text('{"temperature": 0}\n', encoding="utf-8")
        run_directory = self._start(requested_config_path=config_path)

        content = manifest.load_run(run_directory)
        self.assertEqual(
            content["requested_config_file"]["sha256"],
            manifest.sha256_file(config_path),
        )
        self.assertEqual(
            content["requested_config_file"]["path"], str(config_path.resolve())
        )

    def test_strict_mode_rejects_dirty_worktree(self):
        with mock.patch.object(
            manifest,
            "capture_git",
            return_value={
                "commit": "b" * 40,
                "dirty": True,
                "root": str(self.root),
                "branch": "main",
                "changed_paths": ["archivo.py"],
                "error": None,
            },
        ):
            with self.assertRaisesRegex(manifest.ManifestError, "dirty"):
                manifest.start_run(
                    kind="smoke",
                    model_id="organization/model",
                    model_revision=PINNED_REVISION,
                    tokenizer_id="organization/model",
                    tokenizer_revision=PINNED_REVISION,
                    requested_config={},
                    results_root=self.root / "raw",
                    run_id="dirty-run",
                    repo_path=self.root,
                )

    def test_strict_mode_rejects_floating_revisions(self):
        with mock.patch.object(
            manifest,
            "capture_git",
            return_value={
                "commit": "b" * 40,
                "dirty": False,
                "root": str(self.root),
                "branch": "main",
                "changed_paths": [],
                "error": None,
            },
        ):
            with self.assertRaisesRegex(manifest.ManifestError, "commit SHA completo"):
                manifest.start_run(
                    kind="smoke",
                    model_id="organization/model",
                    model_revision="main",
                    tokenizer_id="organization/model",
                    tokenizer_revision="main",
                    requested_config={},
                    results_root=self.root / "raw",
                    run_id="floating-run",
                    repo_path=self.root,
                )

    def test_smoke_exceptions_are_recorded(self):
        run_directory = self._start(
            model_revision="main",
            tokenizer_revision=None,
            allow_dirty=True,
            allow_floating_revision=True,
            allow_missing_git=True,
        )
        content = manifest.load_run(run_directory)
        self.assertFalse(content["policy"]["strict"])
        self.assertEqual(
            content["policy"]["exceptions"],
            ["dirty_worktree", "floating_revision", "missing_git"],
        )

    def test_finalize_hashes_artifacts_and_creates_complete_marker(self):
        run_directory = self._start()
        raw = run_directory / "raw" / "responses.jsonl"
        raw.write_text('{"response":"A"}\n', encoding="utf-8")

        path = manifest.finalize_run(run_directory, status="complete")
        content = manifest.load_run(path.parent)

        self.assertEqual(content["status"], "complete")
        self.assertIsNotNone(content["ended_at"])
        self.assertEqual(content["artifacts"][0]["path"], "raw/responses.jsonl")
        self.assertEqual(content["artifacts"][0]["sha256"], manifest.sha256_file(raw))
        self.assertEqual(len(content["artifacts"]), 1)
        self.assertTrue((run_directory / "COMPLETE").is_file())
        self.assertTrue(raw.is_file())

    def test_failed_finalize_records_error_and_is_not_repeatable(self):
        run_directory = self._start()
        manifest.finalize_run(run_directory, status="failed", errors=["OOM"])

        content = manifest.load_run(run_directory)
        self.assertEqual(content["errors"], [{"message": "OOM"}])
        self.assertTrue((run_directory / "FAILED").is_file())
        with self.assertRaisesRegex(manifest.ManifestError, "ya fue finalizada"):
            manifest.finalize_run(run_directory)

    def test_finalize_marks_failed_when_an_input_changed(self):
        run_directory = self._start()
        self.input_path.write_text("item,value\n1,changed\n", encoding="utf-8")

        with self.assertRaisesRegex(manifest.ManifestError, "inputs cambiaron"):
            manifest.finalize_run(run_directory)

        content = manifest.load_run(run_directory)
        self.assertEqual(content["status"], "failed")
        self.assertFalse(content["input_verification"][0]["matches"])
        self.assertEqual(content["errors"][-1]["type"], "input_integrity_error")
        self.assertTrue((run_directory / "FAILED").is_file())
        self.assertFalse((run_directory / "COMPLETE").exists())

    def test_finalize_revalidates_requested_config_file(self):
        config_path = self.root / "config.json"
        config_path.write_text('{"temperature": 0}\n', encoding="utf-8")
        run_directory = self._start(requested_config_path=config_path)
        config_path.write_text('{"temperature": 1}\n', encoding="utf-8")

        with self.assertRaisesRegex(manifest.ManifestError, "inputs cambiaron"):
            manifest.finalize_run(run_directory)

        content = manifest.load_run(run_directory)
        config_verification = content["input_verification"][-1]
        self.assertEqual(config_verification["name"], "requested_config")
        self.assertFalse(config_verification["matches"])

    def _rewrite_context(self, directory, transform):
        path = directory / "raw/context.json"
        context = json.loads(path.read_text())
        transform(context)
        path.write_text(json.dumps(context))
        manifest_path = directory / "manifest.json"
        compact = json.loads(manifest_path.read_text())
        compact["context"]["sha256"] = manifest.sha256_file(path)
        manifest_path.write_text(json.dumps(compact))

    def test_compact_manifest_contains_only_essential_fields(self):
        directory = self._start()
        core = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(set(core), {
            "schema_version", "run_id", "kind", "status", "started_at", "ended_at",
            "git", "hosts", "model", "errors", "context",
        })
        self.assertEqual(set(core["git"]), {"commit", "dirty"})
        self.assertEqual(core["context"]["sha256"],
                         manifest.sha256_file(directory / "raw/context.json"))

    def test_system_baseline_omits_unrelated_collection(self):
        directory = self._start_system()
        context = json.loads((directory / "raw/context.json").read_text())
        self.assertNotIn("packages", context)
        self.assertNotIn("services", context)
        run = manifest.load_run(directory)
        self.assertEqual(run["packages"], {})
        self.assertEqual(run["services"], {})

    def test_hosts_describe_observation_not_requested_topology(self):
        with mock.patch.object(manifest, "capture_system", return_value={"hostname": "pegasus"}):
            directory = self._start(requested_config={
                "topology": {"hosts": ["pegasus", "octopus"], "num_nodes": 2}
            })
        run = manifest.load_run(directory)
        self.assertEqual(run["hosts"], ["pegasus"])
        self.assertEqual(run["requested_config"]["topology"]["num_nodes"], 2)

    def test_missing_or_changed_context_is_rejected(self):
        directory = self._start()
        context = directory / "raw/context.json"
        original = context.read_bytes()
        context.write_bytes(original + b" ")
        with self.assertRaisesRegex(manifest.ManifestError, "SHA-256"):
            manifest.load_run(directory)
        with self.assertRaisesRegex(manifest.ManifestError, "SHA-256"):
            manifest.finalize_run(directory)
        self.assertFalse((directory / "COMPLETE").exists())
        context.unlink()
        with self.assertRaisesRegex(manifest.ManifestError, "ausente"):
            manifest.load_run(directory)

    def test_context_path_escape_and_external_symlink_are_rejected(self):
        directory = self._start()
        core_path = directory / "manifest.json"
        core = json.loads(core_path.read_text())
        core["context"]["path"] = "../context.json"
        core_path.write_text(json.dumps(core))
        with self.assertRaisesRegex(manifest.ManifestError, "referencia"):
            manifest.load_run(directory)
        core["context"]["path"] = "raw/context.json"
        core_path.write_text(json.dumps(core))
        context = directory / "raw/context.json"
        outside = self.root / "context.json"
        outside.write_bytes(context.read_bytes())
        context.unlink()
        context.symlink_to(outside)
        with self.assertRaisesRegex(manifest.ManifestError, "fuera"):
            manifest.load_run(directory)

    def test_context_identity_and_reserved_fields_are_rejected(self):
        changes = (
            (lambda c: c.update(run_id="another-run"), "identidad"),
            (lambda c: c.update(status="complete"), "inesperados"),
            (lambda c: c["git"].update(commit="c" * 40), "Git"),
            (lambda c: c["system"].update(hostname="octopus"), "hosts"),
            (lambda c: c.pop("inputs"), "incompleto"),
        )
        for index, (transform, message) in enumerate(changes):
            with self.subTest(message=message):
                directory = self._start(run_id=f"context-{index}")
                self._rewrite_context(directory, transform)
                with self.assertRaisesRegex(manifest.ManifestError, message):
                    manifest.load_run(directory)

    def test_interrupted_context_or_manifest_publication_is_not_success(self):
        original_writer = manifest._atomic_write_json
        directory = self._start()
        def fail_manifest_write(path, value):
            if path.name == "manifest.json":
                raise OSError("simulated manifest publication failure")
            return original_writer(path, value)
        with mock.patch.object(manifest, "_atomic_write_json", side_effect=fail_manifest_write):
            with self.assertRaises(OSError):
                manifest.finalize_run(directory)
        self.assertFalse((directory / "COMPLETE").exists())
        with self.assertRaisesRegex(manifest.ManifestError, "SHA-256"):
            manifest.load_run(directory)
        directory = self._start(run_id="interrupted-marker")
        with mock.patch.object(manifest, "_atomic_write_text", side_effect=OSError("marker failure")):
            with self.assertRaises(OSError):
                manifest.finalize_run(directory)
        with self.assertRaisesRegex(manifest.ManifestError, "cierre incompleto"):
            manifest.load_run(directory)

    def test_marker_state_and_date_must_agree(self):
        directory = self._start()
        (directory / "COMPLETE").write_text("premature")
        with self.assertRaisesRegex(manifest.ManifestError, "cierre inconsistente"):
            manifest.load_run(directory)
        (directory / "COMPLETE").unlink()
        manifest.finalize_run(directory)
        (directory / "COMPLETE").write_text("wrong time")
        with self.assertRaisesRegex(manifest.ManifestError, "fecha del marcador"):
            manifest.load_run(directory)
        (directory / "FAILED").write_text("wrong state")
        with self.assertRaisesRegex(manifest.ManifestError, "cierre incompleto"):
            manifest.load_run(directory)

    def test_legacy_read_and_finalize_are_rejected_without_changes(self):
        for index, version in enumerate(("1.0", "1.1")):
            with self.subTest(version=version):
                directory = (self._start if index == 0 else self._start_system)(
                    run_id=f"legacy-{index}")
                legacy = manifest.load_run(directory)
                legacy["schema_version"] = version
                legacy.pop("context")
                legacy.pop("hosts")
                (directory / "manifest.json").write_text(json.dumps(legacy))
                (directory / "raw/context.json").write_text('{"historical": true}')
                before = {path.relative_to(directory): path.read_bytes()
                          for path in directory.rglob("*") if path.is_file()}
                with self.assertRaisesRegex(manifest.ManifestError, "schema_version no soportada"):
                    manifest.load_run(directory)
                for status in ("complete", "failed"):
                    with self.subTest(status=status), self.assertRaisesRegex(
                        manifest.ManifestError, "schema_version no soportada"
                    ):
                        manifest.finalize_run(directory, status=status)
                after = {path.relative_to(directory): path.read_bytes()
                         for path in directory.rglob("*") if path.is_file()}
                self.assertEqual(before, after)
                self.assertFalse((directory / "COMPLETE").exists())
                self.assertFalse((directory / "FAILED").exists())

if __name__ == "__main__":
    unittest.main()
