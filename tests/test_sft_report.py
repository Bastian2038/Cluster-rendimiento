"""Fixtures CPU para reconstruir manifiestos y cierres de SFT."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from infraestructura import manifest, sft_protocol, sft_report, sft_runtime
from infraestructura.serving_capacity_protocol import _hardware_roster


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


class SFTReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / "raw"
        self.anchor = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)
        self.campaign_jobs = {}
        self.run_offsets = {}
        self.config = sft_protocol.load_config()
        self.cluster = sft_protocol.cluster_config(self.config)
        self.git = {"commit": "b" * 40, "dirty": False}
        shard = {"name": "model-00001-of-00001.safetensors", "size_bytes": 16_060_522_496,
                 "blob_name": "e" * 64, "present": True}
        self.model_metadata = {name: _digest(name.encode()) for name in
                               ("config.json", "tokenizer.json", "tokenizer_config.json")}
        self.model_source = {"index_sha256": "f" * 64, "metadata": self.model_metadata,
                             "shards": [{"path": shard["name"], "sha256": shard["blob_name"],
                                         "bytes": shard["size_bytes"]}]}
        model = {"metadata_sha256": {**self.model_metadata, "model.safetensors.index.json": "f" * 64},
                 "shards": [shard], "complete": True}
        nodes = [{"node_id": row["node_id"], "hostname": row["expected_hostname"],
                  "architecture": "aarch64",
                  "models": {self.config["model"]["key"]: model},
                  "gpu": {"returncode": 0, "stdout": f"GPU-fixture-{index}, NVIDIA GB10, 580.1"}}
                 for index, row in enumerate(self.cluster["nodes"])]
        _, self.roster = _hardware_roster({"nodes": nodes})
        history = {"status": "ready_for_smoke", "errors": [], "nodes": nodes}
        history_bytes = (json.dumps(history) + "\n").encode()
        self.config["reference_preflight_sha256"] = _digest(history_bytes)
        runtime = {**{key: self.config["image"][key] for key in (
            "torch_version", "torch_runtime_version", "transformers_version", "tokenizers_version")},
            "python": "3.12.1", "torch_cuda_version": "13.0", "nccl_version": [2, 28, 1],
            "gpus_hidden": True, "nccl_available": True}
        image = {key: self.config["image"][key] for key in ("id", "architecture")}
        observed = [{**row, "sft_container_probe": runtime,
                     "image": {"returncode": 0, "stdout": f"{image['id']} {image['architecture']}"}}
                    for row in nodes]
        self.preflight = {"status": "ready", "errors": [], "roster": self.roster, "nodes": observed,
                          "reference_preflight_sha256": _digest(history_bytes),
                          "removal": self._removal()}
        # En las fixtures solo se sustituye el preflight histórico de hardware.
        recipe = {key: self.config[key] for key in sft_protocol._RECIPE_FIELDS}
        fixture_recipe = _digest(json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        self.enterContext(patch.object(sft_protocol, "_RECIPE_SHA256", fixture_recipe))
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "reference").mkdir()
        (self.inputs / "reference/preflight.json").write_bytes(history_bytes)
        (self.inputs / "corpus.jsonl").write_bytes(sft_protocol.corpus_bytes())
        _write(self.inputs / "requested_config", self.config)
        _write(self.inputs / "cluster.json", self.cluster)
        _write(self.inputs / f"code/{self.config['cluster_config']}", self.cluster)
        for name in sft_report._REQUIRED_INPUTS:
            if name.startswith("code/infraestructura/"):
                path = self.inputs / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# Fixture {name}\n", encoding="utf-8")
        self.input_hashes = {path.relative_to(self.inputs).as_posix(): manifest.sha256_file(path)
                             for path in self.inputs.rglob("*") if path.is_file()}
        captures = {
            "capture_git": self.git, "capture_system": {"hostname": "coordinator"},
            "capture_nvidia": {}, "capture_packages": {},
        }
        for key, value in captures.items():
            mock = patch.object(manifest, key, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def _job(self, stage, group, block=0):
        return next(job for job in sft_protocol.make_plan(self.config, stage)["jobs"]
                    if job["group"] == group and job["block"] == block)

    def _stamp(self, seconds):
        return (self.anchor + timedelta(seconds=seconds)).isoformat()

    def _new_run(self, campaign_id, job):
        run_id = f"{campaign_id}-{job['name']}"
        model = self.config["model"]
        index = self.campaign_jobs.get(campaign_id, 0)
        self.campaign_jobs[campaign_id] = index + 1
        self.run_offsets[run_id] = 60 + index * 200
        with patch.object(manifest, "utc_now", return_value=self._stamp(self.run_offsets[run_id])):
            directory = manifest.start_run(
                kind="full-sft", model_id=model["id"], model_revision=model["revision"],
                tokenizer_id=model["tokenizer_id"], tokenizer_revision=model["tokenizer_revision"],
                requested_config={"campaign_id": campaign_id, "job": job, "protocol": self.config},
                requested_config_path=self.inputs / "requested_config",
                inputs={name: self.inputs / name for name in self.input_hashes if name != "requested_config"},
                results_root=self.raw, run_id=run_id,
            )
        shutil.copytree(self.inputs, directory / "raw/inputs")
        workspace = str(Path(self.config["paths"]["workspace"]) / campaign_id / run_id)
        staged = {"config.json": self.input_hashes["requested_config"], "corpus.jsonl": self.input_hashes["corpus.jsonl"]}
        staged.update({name: digest for name, digest in self.input_hashes.items() if name.startswith("code/")})
        _write(directory / "raw/workspace.json",
               {"retained": True, "path": workspace, "errors": {}, "setup_before_launch_seconds": 2,
                "staged": {node: {"workspace": workspace, "uid": 1000, "files": staged}
                           for node in job["nodes"]}})
        return directory

    def _identity(self, directory, node, rank, job, campaign_id, phase):
        fields = ("hostname", "gpu_uuid")
        identity = {key: self.roster[node][key] for key in fields}
        identity.update(campaign_id=campaign_id, run_id=directory.name, node_id=node, phase=phase,
                        rank=rank, world_size=len(job["nodes"]))
        return identity

    def _phase_roster(self, directory, job, campaign_id, phase):
        roster = [self._identity(directory, node, rank, job, campaign_id, phase)
                  for rank, node in enumerate(job["nodes"])]
        participants = []
        for identity in roster:
            relative = f"raw/nodes/{identity['node_id']}/{phase}"
            _write(directory / relative / "identity.json", identity)
            _write(directory / relative / "group.json", {**identity, "backend": "nccl", "participants": roster})
            participants.append({"evidence_path": f"{relative}/identity.json",
                                 **{key: identity[key] for key in ("node_id", "hostname", "gpu_uuid", "rank")}})
        if phase in ("pilot", "measure"):
            manifest.register_participants(directory, participants)
        return roster

    def _write_phase(self, directory, job, campaign_id, phase, *, seconds=1, loss_offset=0):
        ranges = {"pilot": range(1, 5), "resume": range(3, 5), "measure": range(1, 26)}
        roster = self._phase_roster(directory, job, campaign_id, phase)
        for identity in roster:
            node, rank = identity["node_id"], identity["rank"]
            base = directory / f"raw/nodes/{node}/{phase}"
            rows = []
            cuda = {"allocated_bytes": 10_000, "reserved_bytes": 15_000, "peak_allocated_bytes": 12_000,
                    "device_free_bytes": 50_000, "device_total_bytes": 65_000}
            for number in ranges[phase]:
                ids = sft_protocol.step_ids(number - 1, rank, len(job["nodes"]))
                rows.append({"step": number, "ids": ids, "seconds": seconds + rank / 10,
                             "processed_tokens": 512 * len(ids), "supervised_tokens": 128 * len(ids),
                             "loss": 3 / number + loss_offset, "loss_definition": "global_supervised_token_mean",
                             "cuda_memory": cuda, "rank": rank, "node_id": node, "phase": phase,
                             "measured": phase == "measure" and number > 5,
                             "audit_performed": phase == "pilot" and number == 4,
                             "parameter_audit_seconds": 20 if phase == "pilot" and number == 4 else 0,
                             "global_processed_tokens": 2048, "global_supervised_tokens": 512})
            _write_rows(base / "steps.jsonl", rows)
            gpu = {"uuid": identity["gpu_uuid"], "temperature_gpu": 42 + rank, "power_draw": None,
                   "utilization_gpu": 90, "memory_used": None, "memory_total": None,
                   "clocks_sm": 1000, "clocks_mem": 2000}
            def sample(stage, instant):
                return {**identity, "training_stage": stage, "monotonic_ns": instant,
                        "timestamp_utc": self._stamp(self.run_offsets[directory.name] + instant / 1e9),
                        "gpus": [gpu], "cuda_memory": cuda,
                        "memory": {"MemAvailable_kib": 100_000}}
            telemetry = [sample("startup", 1_000_000_000)]
            start = 9_000_000_000
            window = 21 * (seconds + rank / 10)
            end = start + round(window * 1e9)
            if phase == "measure":
                telemetry += [sample("warmup", start - 1000), sample("measured", start + 1000),
                              sample("measured", end - 1000), sample("validation", end + 1000)]
            else:
                telemetry.append(sample(phase, 2_000_000_000))
            _write_rows(base / "telemetry.jsonl", telemetry)
            fixed = {key: self.config["workload"][key] for key in ("corpus_sha256", "tokens_sha256")}
            versions = {key: self.config["image"][key] for key in (
                "torch_version", "torch_runtime_version", "transformers_version")}
            model = self.config["model"]
            artifact_records = {}
            for name in ("identity.json", "group.json", "steps.jsonl", "telemetry.jsonl"):
                path = base / name
                artifact_records[name] = {"sha256": manifest.sha256_file(path), "bytes": path.stat().st_size}
            training_fields = ("global_batch", "microbatch", "compute_dtype", "master_dtype",
                               "gradient_dtype", "use_cache", "optimizer", "learning_rate")
            training = {key: self.config["training"][key] for key in training_fields}
            receipt = {**identity, **fixed, **versions, **training,
                       "phase_seconds": 20 + 35 * (seconds + rank / 10),
                       "model_revision": model["revision"], "tensor_count": model["tensor_count"],
                       "tensor_shapes_sha256": model["tensor_shapes_sha256"],
                       "parameter_count": model["parameter_count"], "peak_allocated_bytes": 24 * 1024**3,
                       "window_seconds": window if phase == "measure" else None, "startup_seconds": 10 + rank,
                       "window_start_monotonic_ns": start if phase == "measure" else None,
                       "window_end_monotonic_ns": end if phase == "measure" else None,
                       "tokenizer_revision": model["tokenizer_revision"], "model_source": self.model_source,
                       "status": "ok", "schema_version": "1.0", "config_sha256": self.input_hashes["requested_config"],
                       "artifacts": artifact_records, "participants": roster, "reduce_dtype": "float32",
                       "startup_scope": "worker_execute_to_first_step", "attention": "sdpa",
                       "activation_checkpointing": "non-reentrant",
                       "gradient_accumulation_steps": 4 // len(roster), "checkpoint": None,
                       "proof": {"audit_performed": False, "tensor_count": model["tensor_count"],
                                 "parameter_count": model["parameter_count"]}}
            if phase in ("pilot", "resume"):
                receipt.update(self._pilot_receipt(phase, rank, len(job["nodes"])))
            _write(base / "receipt.json", receipt)
    def _removal(self, removed=None, *, time_offset=10):
        nodes = self.cluster["groups"]["four"]
        ids = removed or {}
        return {"removal_confirmed": True, "ended_at": self._stamp(time_offset),
                "nodes": {node: {"confirmed": True, "remaining_ids": [], "removed_ids": ids.get(node, [])}
                          for node in nodes}}

    def _cleanup(self, directory, campaign_id, job, *, oom=False):
        phases = ("pilot",) if oom else ("pilot", "resume") if job["block"] == 0 else ("measure",)
        receipts = {"final": self._removal()}
        for phase in phases:
            containers = {node: _digest(f"{phase}/{node}".encode()) for node in job["nodes"]}
            receipts[phase] = removal = self._removal({node: [cid] for node, cid in containers.items()})
            states = {node: {"Status": "exited", "ExitCode": 1 if oom else 0, "OOMKilled": False, "Error": ""}
                      for node in job["nodes"]}
            commands = {}
            resources = {}
            workspace = str(Path(self.config["paths"]["workspace"]) / campaign_id / directory.name)
            limits = sft_runtime.resource_limits(self.config, [{"node_id": node} for node in job["nodes"]], [job])
            for rank, node in enumerate(job["nodes"]):
                worker = sft_protocol.worker_argv(
                    self.config, job, rank, phase, config_path="/work/config.json",
                    model_dir=Path("/models") / sft_protocol.model_subdir(self.config), out=f"/work/out/{phase}",
                    checkpoint_dir="/work/checkpoint", campaign_id=campaign_id, run_id=directory.name,
                )
                network = self.config["network"]["environment"]
                commands[node] = ["docker", "create", "--pull=never", self.config["image"]["id"],
                                  *[f"{key}={value}" for key, value in network.items()], *worker]
                host = {"memory_bytes": {"MemAvailable": 100 * sft_runtime.GIB},
                        "resources": {"cpu_affinity_count": 8, "workspace_writable": True, "workspace": workspace,
                                      "checkpoint_disk_free_inodes": 10_000,
                                      "checkpoint_disk_free_bytes": 100 * sft_runtime.GIB},
                        "sft_container_probe": {"memory_max": "max", "cpu_max": "max 100000", "cpu_affinity_count": 8}}
                limit = limits[node]
                if phase != "pilot":
                    limit.update(checkpoint_bytes=0, checkpoint_inodes=0)
                resources[node] = {"node_id": node, "phase": phase,
                                   "observed_at": self._stamp(self.run_offsets[directory.name] + 2),
                                   "memory_bytes": host["memory_bytes"], "resources": host["resources"],
                                   "assessment": sft_runtime._check_resources(host, limit)}
                _write(directory / f"logs/{phase}/{node}.json",
                       {"command": ["docker", "logs", "--tail", "5000", containers[node]], "returncode": 0,
                        "timed_out": False, "stdout": "NCCL INFO Init COMPLETE", "stderr": "",
                        "duration_seconds": 0.01})
            launch_seconds = 77 if job["group"] == "four" else 50
            _write(directory / f"logs/{phase}-terminal.json",
                   {"removal": removal, "errors": [], "states": states, "launch_to_terminal_seconds": launch_seconds})
            _write(directory / f"logs/{phase}-commands.json", commands)
            _write(directory / f"logs/{phase}-containers.json", containers)
            _write(directory / f"logs/{phase}-resources.json", resources)
        _write(directory / "raw/cleanup.json",
               {"campaign_id": campaign_id, "run_id": directory.name, "removal_confirmed": True,
                "removal_seconds": 4, "coverage": self.cluster["groups"]["four"], "phases": receipts,
                "nodes": [{"node_id": node, "status": "confirmed_absent"} for node in job["nodes"]]})

    def _finish(self, directory, job, outcome, *, status="complete"):
        errors = [{"type": "cuda_oom", "message": "OOM CUDA documentado por un rank"}] if outcome == "oom" else []
        with patch.object(manifest, "utc_now", return_value=self._stamp(self.run_offsets[directory.name] + 180)):
            manifest.finalize_run(directory, status=status, errors=errors)
        return {"job": job["name"], "group": job["group"], "block": job["block"], "run_id": directory.name,
                "status": status, "outcome": outcome, "duration_seconds": 180,
                "manifest_sha256": manifest.sha256_file(directory / "manifest.json")}

    def _close(self, campaign_id, stage, refs, qualified, *, status="PARTIAL", pilot_reference=None):
        directory = self.raw / "sft-campaign" / campaign_id
        if pilot_reference is not None:
            pilot = sft_report.build_report(directory.parent / pilot_reference["campaign_id"])
            published = directory / "published-pilot.json"
            _write(published, pilot)
            pilot_reference = {**pilot_reference, "published_report": {
                "path": f"docs/resultados/sft-piloto-{pilot['campaign_id']}.json",
                "sha256": manifest.sha256_file(published), "git_commit": self.git["commit"],
            }}
        campaign = {"campaign_id": campaign_id, "stage": stage, "status": status.lower(), "git": self.git,
                    "started_at": self._stamp(0), "ended_at": self._stamp(2000),
                    "duration_seconds": 2000, "limit_seconds": self.config["measurement"][f"{stage}_limit_seconds"],
                    "removal_confirmed": True,
                    "runs": refs, "errors": [], "qualified_groups": qualified, "input_hashes": self.input_hashes,
                    "pilot_reference": pilot_reference}
        plan = sft_protocol.make_plan(self.config, stage, qualified if stage == "run" else None)
        documents = {"plan": plan, "cluster": self.cluster, "config": self.config,
                     "preflight": self.preflight, "campaign": campaign}
        hashes = {}
        for name, payload in documents.items():
            file = directory / f"{name}.json"
            _write(file, payload)
            hashes[f"{name}_sha256"] = manifest.sha256_file(file)
        shutil.copytree(self.inputs, directory / "inputs", ignore=shutil.ignore_patterns("requested_config"))
        cleanup = directory / "cleanup.json"
        _write(cleanup, self._removal(time_offset=1999))
        _write(directory / status, {**hashes, "cleanup_sha256": manifest.sha256_file(cleanup),
                                    "sealed_at": self._stamp(2000.25), "duration_seconds": 2000.25})
        self._publish(directory, status)
        return directory

    def _publish(self, directory, status, *, observed=2000.5, elapsed=None):
        _write(directory / "publication.json",
               {"campaign_id": directory.name, "status": status.lower(),
                "marker_sha256": manifest.sha256_file(directory / status),
                "observed_at": self._stamp(observed), "duration_seconds": observed if elapsed is None else elapsed})

    def _pilot_receipt(self, phase, rank, world):
        model = self.config["model"]
        files = []
        totals, hashes = [], []
        for index, (name, shape) in enumerate(sft_protocol.llama_tensor_shapes().items()):
            size = math.prod(shape)
            local_shape = [shape[0] // world, *shape[1:]]
            local_size = math.prod(local_shape)
            files.append({"name": name, "path": f"tensor-{index:04d}.pt", "shape": local_shape,
                          "global_shape": shape, "dtype": "torch.float32", "bytes": 4 * local_size,
                          "sha256": _digest(f"{rank}/{index}".encode())})
            totals.append({"name": name, "numel": size, "finite_grad_elements": size,
                           "nonzero_grad_elements": size - 1, "changed_shards": 1})
            before = _digest(f"{rank}/{index}/before".encode())
            after = _digest(f"{rank}/{index}/after".encode()) if rank == index % world else before
            hashes.append({"name": name, "before_sha256": before, "after_sha256": after, "local_numel": local_size,
                           "finite_grad_elements": local_size, "nonzero_grad_elements": local_size - int(rank == 0)})
        files.append({"path": "state.pt", "bytes": 2048, "sha256": _digest(f"{rank}/state".encode())})
        restored = phase == "resume"
        proof = {"audit_performed": not restored, "tensor_count": model["tensor_count"],
                 "parameter_count": model["parameter_count"]}
        if not restored:
            proof.update(audited_step=4, gradient_finite_tensor_count=model["tensor_count"],
                         gradient_nonzero_tensor_count=model["tensor_count"], updated_tensor_count=model["tensor_count"],
                         method="sha256-fp32-local-before-after-sgd", hash_chunk_bytes=1048576,
                         tensors=totals, local_weight_hashes=hashes)
        return {"proof": proof, "checkpoint": {
            "metadata_sha256": _digest(f"{rank}/metadata".encode()), "step": 2, "next_step": 3,
            "restored": restored, "rng_restored": restored, "hashes_verified": True, "files": files}}

    def _pilot(self, groups=("four", "pair")):
        refs = []
        for group in ("four", "pair", "single"):
            job = self._job("pilot", group)
            run = self._new_run("pilot-a", job)
            if group in groups:
                self._write_phase(run, job, "pilot-a", "pilot")
                self._write_phase(run, job, "pilot-a", "resume")
                self._cleanup(run, "pilot-a", job)
                refs.append(self._finish(run, job, "qualified"))
            else:
                self._oom(run, job, "pilot-a")
                refs.append(self._finish(run, job, "oom", status="failed"))
        pilot = self._close("pilot-a", "pilot", refs, list(groups), status="COMPLETE")
        report = sft_report.build_report(pilot)
        return pilot, {"campaign_id": pilot.name, "campaign_sha256": manifest.sha256_file(pilot / "campaign.json"),
                       "git_commit": self.git["commit"], "estimated_matrix_seconds": report["estimated_matrix_seconds"]}

    def _oom(self, run, job, campaign_id):
        for identity in self._phase_roster(run, job, campaign_id, "pilot"):
            node = identity["node_id"]
            directory = run / f"raw/nodes/{node}/pilot"
            _write(directory / "failure.json",
                   {**identity, "type": "cuda_oom", "status": "cuda_oom", "message": "CUDA out of memory",
                    "exception_type": "torch.OutOfMemoryError", "stage": "train_step", "oom_origin": "cuda",
                    "timestamp_utc": self._stamp(self.run_offsets[run.name] + 45)})
        self._cleanup(run, campaign_id, job, oom=True)

    def test_qualification_reconstructs_checkpoint_and_projects_three_blocks(self):
        pilot, _ = self._pilot()
        report = sft_report.build_report(pilot)
        self.assertEqual(report["ended_at"], self._stamp(2000.5))
        self.assertEqual(report["duration_seconds"], 2000.5)
        self.assertEqual(report["index_ended_at"], self._stamp(2000))
        self.assertEqual(report["sealed_duration_seconds"], 2000.25)
        self.assertEqual(report["publication_sha256"], manifest.sha256_file(pilot / "publication.json"))
        self.assertEqual(report["qualified_groups"], ["four", "pair"])
        four = report["runs"][0]
        self.assertEqual(four["pilot_step_seconds"], [1.3] * 4)
        self.assertEqual(four["phase_steps"]["pilot"][3]["parameter_audit_seconds"], 20)
        self.assertEqual([row["step"] for row in four["restoration"]], [3, 4])
        overhead = 77 - (20 + 35 * 1.3)
        self.assertAlmostEqual(four["startup_components"]["launcher_overhead_seconds"], overhead)
        self.assertEqual(report["runs"][1]["startup_components"]["launcher_overhead_seconds"], 0)
        expected = 3 * 1.25 * ((2 + overhead + 13 + 25 * 1.3 + 4) + (2 + 11 + 25 * 1.1 + 4))
        self.assertAlmostEqual(report["estimated_matrix_seconds"], expected)
        self.assertEqual(report["pending_jobs"], [])

    def test_pilot_rejects_loss_outside_tolerance_before_close(self):
        job = self._job("pilot", "single")
        run = self._new_run("pilot-loss", job)
        self._write_phase(run, job, "pilot-loss", "pilot")
        self._write_phase(run, job, "pilot-loss", "resume", loss_offset=0.01)
        self._cleanup(run, "pilot-loss", job)
        assessment = sft_report.assess_run(run, job, self.config)
        self.assertEqual(assessment["outcome"], "indeterminate")
        self.assertIn("fuera de tolerancia", assessment["issues"][0])

    def test_pilot_recounts_update_hashes_across_ranks(self):
        job = self._job("pilot", "pair")
        run = self._new_run("pilot-proof", job)
        self._write_phase(run, job, "pilot-proof", "pilot")
        self._write_phase(run, job, "pilot-proof", "resume")
        self._cleanup(run, "pilot-proof", job)
        path = run / "raw/nodes/pegasus/pilot/receipt.json"
        receipt = json.loads(path.read_text())
        changed = receipt["proof"]["local_weight_hashes"][0]
        original = dict(changed)
        changed["after_sha256"] = changed["before_sha256"]
        _write(path, receipt)
        result = sft_report.assess_run(run, job, self.config)
        self.assertEqual(result["outcome"], "indeterminate")
        self.assertIn("changed_shards incompatible", result["issues"][0])
        changed.update(original)
        changed["nonzero_grad_elements"] += 1
        _write(path, receipt)
        result = sft_report.assess_run(run, job, self.config)
        self.assertIn("nonzero_grad_elements incompatible", result["issues"][0])

    def test_pilot_rejects_renamed_layout_with_identical_totals(self):
        job = self._job("pilot", "single")
        run = self._new_run("pilot-layout", job)
        self._write_phase(run, job, "pilot-layout", "pilot")
        self._write_phase(run, job, "pilot-layout", "resume")
        self._cleanup(run, "pilot-layout", job)
        for phase in ("pilot", "resume"):
            path = run / f"raw/nodes/pegasus/{phase}/receipt.json"
            receipt = json.loads(path.read_text())
            receipt["checkpoint"]["files"][0]["name"] = "fixture.renamed.weight"
            if phase == "pilot":
                for key in ("tensors", "local_weight_hashes"):
                    receipt["proof"][key][0]["name"] = "fixture.renamed.weight"
            _write(path, receipt)
        result = sft_report.assess_run(run, job, self.config)
        self.assertIn("layout BF16 fijado", result["issues"][0])

    def test_success_requires_torch_versions_resources_and_log_for_each_rank(self):
        job = self._job("run", "pair", 1)
        run = self._new_run("matrix-evidence", job)
        self._write_phase(run, job, "matrix-evidence", "measure")
        self._cleanup(run, "matrix-evidence", job)
        path = run / "raw/nodes/octopus/measure/receipt.json"
        receipt = json.loads(path.read_text())
        _write(path, {**receipt, "torch_runtime_version": receipt["torch_version"]})
        self.assertIn("torch_runtime_version incompatible", sft_report.assess_run(run, job, self.config)["issues"][0])
        _write(path, receipt)
        path = run / "logs/measure-resources.json"
        resources = json.loads(path.read_text())
        _write(path, {"pegasus": resources["pegasus"]})
        self.assertIn("recursos", sft_report.assess_run(run, job, self.config)["issues"][0])
        _write(path, resources)
        log_path = run / "logs/measure/octopus.json"
        log = json.loads(log_path.read_text())
        _write(log_path, {**log, "command": [*log["command"][:-1], "a" * 64]})
        self.assertIn("logs de otro contenedor", sft_report.assess_run(run, job, self.config)["issues"][0])

    def test_requires_explicit_oom_and_preserves_failed_run(self):
        job = self._job("pilot", "single")
        run = self._new_run("pilot-failure", job)
        self.assertEqual(sft_report.assess_run(run, job, self.config)["outcome"], "indeterminate")
        self._oom(run, job, "pilot-failure")
        path = run / "raw/nodes/pegasus/pilot/failure.json"
        failure = json.loads(path.read_text())
        _write(path, {**failure, "message": "fixture CPU"})
        self.assertEqual(sft_report.assess_run(run, job, self.config)["outcome"], "indeterminate")
        _write(path, failure)
        terminal_path = run / "logs/pilot-terminal.json"
        terminal = json.loads(terminal_path.read_text())
        killed = {**terminal["states"]["pegasus"], "ExitCode": 137, "OOMKilled": True}
        _write(terminal_path, {**terminal, "states": {"pegasus": killed}})
        self.assertEqual(sft_report.assess_run(run, job, self.config)["outcome"], "indeterminate")
        _write(terminal_path, terminal)
        ref = self._finish(run, job, "oom", status="failed")
        campaign = self._close("pilot-failure", "pilot", [ref], [])
        assessment = sft_report.build_report(campaign)["runs"][0]
        self.assertEqual(assessment["outcome"], "oom")
        self.assertIsNone(assessment["window_seconds"])
        self.assertEqual(len(assessment["oom_evidence"]), 1)
        self.assertIn("pegasus, fase `pilot`, etapa `train_step`", sft_report._markdown(sft_report.build_report(campaign)))

    def test_unattributable_oom_preserves_observation_in_partial(self):
        job = self._job("pilot", "pair")
        run = self._new_run("pilot-no-group", job)
        self._oom(run, job, "pilot-no-group")
        base = run / "raw/nodes/octopus/pilot"
        group = json.loads((base / "group.json").read_text())
        failure = json.loads((base / "failure.json").read_text())
        _write(base / "failure.json", {**failure, "stage": "parameter_audit"})
        result = sft_report.assess_run(run, job, self.config)
        self.assertEqual(result["outcome"], "indeterminate")
        self.assertEqual(result["oom_evidence"][1]["stage"], "parameter_audit")
        _write(base / "failure.json", {**failure, "stage": "group_init"})
        result = sft_report.assess_run(run, job, self.config)
        self.assertEqual(result["outcome"], "indeterminate")
        self.assertEqual(result["oom_evidence"][1]["stage"], "group_init")
        _write(base / "failure.json", failure)
        (base / "group.json").unlink()
        result = sft_report.assess_run(run, job, self.config)
        self.assertEqual(result["outcome"], "indeterminate")
        self.assertEqual(len(result["oom_evidence"]), 2)
        _write(base / "group.json", {**group, "participants": []})
        ref = self._finish(run, job, "indeterminate", status="failed")
        campaign = self._close("pilot-no-group", "pilot", [ref], [])
        report = sft_report.build_report(campaign)
        self.assertEqual(report["capacity"]["pair"]["oom"], [])
        self.assertEqual(len(report["runs"][0]["oom_evidence"]), 2)
        self.assertIn("grupo en octopus/pilot", report["runs"][0]["issues"][0])

    def test_partial_matrix_pairs_blocks_and_excludes_corrupt_artifacts(self):
        _, reference = self._pilot()
        refs = []
        runs = {}
        for block, group, seconds in ((1, "four", 1), (1, "pair", 2), (2, "four", 2), (2, "pair", 3)):
            job = self._job("run", group, block)
            run = self._new_run("matrix", job)
            self._write_phase(run, job, "matrix", "measure", seconds=seconds)
            self._cleanup(run, "matrix", job)
            refs.append(self._finish(run, job, "measured"))
            runs[block, group] = run
        step_file = runs[2, "pair"] / "raw/nodes/pegasus/measure/steps.jsonl"
        step_file.write_text("{}", encoding="utf-8")
        campaign = self._close("matrix", "run", refs, ["four", "pair"], pilot_reference=reference)
        report = sft_report.build_report(campaign)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["runs"][0]["step_seconds"], [1.3] * 20)
        self.assertAlmostEqual(report["runs"][0]["processed_tokens_per_second"], 20 * 4 * 512 / (21 * 1.3))
        pair = report["comparisons"]["4/2"]
        self.assertEqual(pair["coverage"], {"paired_blocks": 1, "planned_blocks": 3, "design_blocks": 3})
        self.assertAlmostEqual(pair["speedup"]["median"], 2.1 / 1.3)
        self.assertAlmostEqual(pair["efficiency"]["median"], 2.1 / 2.6)
        self.assertIsNone(report["comparisons"]["2/1"]["speedup"]["median"])
        self.assertAlmostEqual(report["drift"]["four"]["first_to_last_relative_change"], 2.3 / 1.3 - 1)
        self.assertEqual([row["job"] for row in report["failures"]], ["b2-pair"])
        text = sft_report._markdown(report)
        self.assertIn("Pico CUDA asignado por nodo (GiB)", text)
        self.assertIn("s/actualización", text)
        path = campaign / "PARTIAL"
        seal = json.loads(path.read_text())
        _write(path, {**seal, "duration_seconds": 2005})
        self._publish(campaign, "PARTIAL")
        blocked = sft_report.build_report(campaign)
        self.assertIsNone(blocked["comparisons"]["4/2"]["speedup"]["median"])
        self.assertEqual(blocked["runs"][0]["observed_outcome"], "measured")
        self.assertIn("presupuesto de campaña no acreditado", blocked["errors"][0])
        limit = report["limit_seconds"]
        _write(path, {**seal, "duration_seconds": limit - 0.25, "sealed_at": self._stamp(limit - 0.25)})
        self._publish(campaign, "PARTIAL", observed=limit + 0.25)
        late = sft_report.build_report(campaign)
        self.assertIn("duración hasta la publicación superior al límite", late["errors"][0])
        self.assertEqual(late["duration_seconds"], limit + 0.25)
        self.assertEqual(late["runs"][0]["observed_outcome"], "measured")
        self.assertEqual(late["comparisons"]["4/2"]["coverage"]["paired_blocks"], 0)

    def test_matrix_pairs_all_topologies_after_publishing_pilot(self):
        pilot, reference = self._pilot(("four", "pair", "single"))
        self.git["commit"] = "c" * 40
        refs = []
        for block in (1, 2, 3):
            for group, seconds in (("four", 1), ("pair", 2), ("single", 4)):
                job = self._job("run", group, block)
                run = self._new_run("matrix-all", job)
                self._write_phase(run, job, "matrix-all", "measure", seconds=seconds)
                self._cleanup(run, "matrix-all", job)
                refs.append(self._finish(run, job, "measured"))
        campaign = self._close("matrix-all", "run", refs, ["four", "pair", "single"],
                               status="COMPLETE", pilot_reference=reference)
        report = sft_report.build_report(campaign)
        for name, expected in (("4/2", 2.1 / 1.3), ("2/1", 4 / 2.1), ("4/1", 4 / 1.3)):
            comparison = report["comparisons"][name]
            self.assertAlmostEqual(comparison["speedup"]["median"], expected)
            self.assertAlmostEqual(comparison["speedup"]["range"][0], expected)
            self.assertEqual(comparison["coverage"]["paired_blocks"], 3)
        self.assertNotIn("### Exclusiones", sft_report._markdown(report))
        seal_path = campaign / "COMPLETE"
        index_path = campaign / "campaign.json"
        seal = json.loads(seal_path.read_text())
        index = json.loads(index_path.read_text())
        limit = report["limit_seconds"]
        _write(seal_path, {**seal, "duration_seconds": limit - 0.25, "sealed_at": self._stamp(limit - 0.25)})
        self._publish(campaign, "COMPLETE", observed=limit + 0.25, elapsed=limit)
        with self.assertRaisesRegex(ValueError, "duración hasta la publicación superior al límite"):
            sft_report.build_report(campaign)
        _write(seal_path, seal)
        self._publish(campaign, "COMPLETE")
        pilot_receipt = pilot / "publication.json"
        receipt_bytes = pilot_receipt.read_bytes()
        pilot_receipt.write_bytes(receipt_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "derivado publicado difiere del piloto"):
            sft_report.build_report(campaign)
        pilot_receipt.write_bytes(receipt_bytes)
        published_path = campaign / "published-pilot.json"
        published = json.loads(published_path.read_text())
        _write(published_path, {**published, "qualified_groups": ["four"]})
        index["pilot_reference"]["published_report"]["sha256"] = manifest.sha256_file(published_path)
        _write(index_path, index)
        _write(seal_path, {**seal, "campaign_sha256": manifest.sha256_file(index_path)})
        self._publish(campaign, "COMPLETE")
        with self.assertRaisesRegex(ValueError, "derivado publicado difiere del piloto"):
            sft_report.build_report(campaign)

    def test_measurement_rejects_short_window_and_unrelated_telemetry(self):
        job = self._job("run", "single", 1)
        run = self._new_run("bad-measurement", job)
        self._write_phase(run, job, "bad-measurement", "measure")
        self._cleanup(run, "bad-measurement", job)
        base = run / "raw/nodes/pegasus/measure"
        receipt_path = base / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        short = {**receipt, "window_seconds": 19,
                 "window_end_monotonic_ns": receipt["window_start_monotonic_ns"] + 19_000_000_000}
        path = base / "telemetry.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [row for row in rows if row["training_stage"] != "measured"
                or row["monotonic_ns"] < short["window_end_monotonic_ns"]]
        _write_rows(path, rows)
        record = {"sha256": manifest.sha256_file(path), "bytes": path.stat().st_size}
        short["artifacts"] = {**receipt["artifacts"], "telemetry.jsonl": record}
        _write(receipt_path, short)
        result = sft_report.assess_run(run, job, self.config)
        self.assertIn("suma de pasos locales", result["issues"][0])
        rows[2]["gpu_uuid"] = "GPU-fixture-wrong"
        _write_rows(path, rows)
        short["artifacts"]["telemetry.jsonl"] = {"sha256": manifest.sha256_file(path), "bytes": path.stat().st_size}
        _write(receipt_path, short)
        result = sft_report.assess_run(run, job, self.config)
        self.assertIn("identidad de telemetría", result["issues"][0])

    def test_rejects_campaign_marker_drift(self):
        campaign, _ = self._pilot(())
        _write(campaign / "plan.json", {"jobs": []})
        with self.assertRaisesRegex(ValueError, "SHA-256.*plan"):
            sft_report.build_report(campaign)

    def test_complete_checks_cleanup_hash_and_budget_at_seal(self):
        campaign, _ = self._pilot(())
        marker = campaign / "COMPLETE"
        seal = json.loads(marker.read_text())
        _write(marker, {**seal, "sealed_at": self._stamp(7200.25), "duration_seconds": 7200.25})
        self._publish(campaign, "COMPLETE", observed=7200.5)
        with self.assertRaisesRegex(ValueError, "hasta el sello superior al límite"):
            sft_report.build_report(campaign)
        _write(marker, seal)
        self._publish(campaign, "COMPLETE")
        _write(campaign / "cleanup.json", self._removal(time_offset=1998))
        with self.assertRaisesRegex(ValueError, "SHA-256.*cleanup.json"):
            sft_report.build_report(campaign)

    def test_complete_pilot_checks_postlink_observation_and_rejects_bad_receipts(self):
        campaign, _ = self._pilot(())
        path = campaign / "publication.json"
        publication = json.loads(path.read_text())
        path.unlink()
        with self.assertRaisesRegex(ValueError, "publication.json"):
            sft_report.build_report(campaign)
        for field, value in (("status", "partial"), ("campaign_id", "otro-piloto"), ("marker_sha256", "a" * 64)):
            with self.subTest(field=field):
                _write(path, {**publication, field: value})
                with self.assertRaisesRegex(ValueError, f"publication.json.*{field}"):
                    sft_report.build_report(campaign)
        for observed, elapsed, reason in (
            (2000.24, 2000.5, "precede a la muestra del sello"),
            (2000.5, 2000.24, "precede a la muestra del sello"),
            (2000.5, 2003, "relojes UTC y monotónico incompatibles"),
        ):
            with self.subTest(observed=observed, elapsed=elapsed):
                self._publish(campaign, "COMPLETE", observed=observed, elapsed=elapsed)
                with self.assertRaisesRegex(ValueError, reason):
                    sft_report.build_report(campaign)
        marker = campaign / "COMPLETE"
        seal = json.loads(marker.read_text())
        _write(marker, {**seal, "duration_seconds": 7199.75, "sealed_at": self._stamp(7199.75)})
        self._publish(campaign, "COMPLETE", observed=7200, elapsed=7200.25)
        with self.assertRaisesRegex(ValueError, "duración hasta la publicación superior al límite"):
            sft_report.build_report(campaign)

    def test_report_files_are_exclusive_and_outside_evidence(self):
        pilot, _ = self._pilot(())
        report = sft_report.build_report(pilot)
        directory = self.root / "output/raw"
        destination = directory / "report.md"
        directory.mkdir(parents=True)
        destination.write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            sft_report.write_report(destination, report)
        self.assertFalse(destination.with_suffix(".json").exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "original")
        with self.assertRaisesRegex(ValueError, "fuera de raw"):
            sft_report.write_report(pilot / "observations.md", report)
        output = directory / "second.md"
        sft_report.write_report(output, report)
        self.assertIn("SFT completo", output.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(output.with_suffix(".json").read_text(encoding="utf-8")), report)
        basename = "sft-piloto-20261123T000000.123456Z-01234567"
        sft_report.write_report(directory / basename, report)
        self.assertTrue((directory / (basename + ".md")).is_file())
        self.assertEqual(json.loads((directory / (basename + ".json")).read_bytes()), report)
