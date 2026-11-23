import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from infraestructura import inference, manifest, sft_protocol as protocol, sft_runner as runner, sft_runtime as runtime


GB10_SUMMARY = """Sun Nov 22 21:58:24 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   45C    P8              4W /  N/A  | Not Supported          |      {util}%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
{processes}
+-----------------------------------------------------------------------------------------+
"""
DESKTOP_ROWS = """|    0   N/A  N/A            2868      G   /usr/lib/xorg/Xorg                       43MiB |
|    0   N/A  N/A            3018      G   /usr/bin/gnome-shell                     24MiB |"""
EMPTY_PROCESSES = "|  No running processes found                                                             |"


class SftGpuReadinessTests(unittest.TestCase):
    def record(self, rows=DESKTOP_ROWS, util=0):
        return {"gpu_apps": {"returncode": 0, "stdout": "", "stderr": ""},
                "gpu_summary": {"returncode": 0, "stdout": GB10_SUMMARY.format(util=util, processes=rows),
                                "stderr": ""}}

    def test_real_format_admits_only_idle_desktop_or_explicitly_empty_table(self):
        for rows in (DESKTOP_ROWS, EMPTY_PROCESSES, DESKTOP_ROWS.replace("43MiB", "N/A")):
            with self.subTest(rows=rows):
                diagnosis = runtime.gpu_idle_diagnosis(self.record(rows))
                self.assertEqual(diagnosis["status"], "idle")
                self.assertEqual(diagnosis["gpu_utilization_percent"], 0)
                self.assertTrue(diagnosis["compute_apps_empty"])
                processes = diagnosis["graphics_processes"]
                self.assertEqual(len(processes), 0 if rows == EMPTY_PROCESSES else 2)
                self.assertTrue(all(row["type"] == "G" for row in processes))
                if rows != EMPTY_PROCESSES:
                    self.assertEqual([row["executable"] for row in processes],
                                     ["/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell"])

    def test_process_allowlist_requires_exact_path_and_exact_graphics_type(self):
        for changed in (DESKTOP_ROWS.replace(" G ", " C "), DESKTOP_ROWS.replace(" G ", " C+G "),
                        DESKTOP_ROWS.replace(" G ", " g "), DESKTOP_ROWS.replace(" G ", " X "),
                        DESKTOP_ROWS.replace("/usr/lib/xorg/Xorg", "/usr/libexec/Xorg"),
                        DESKTOP_ROWS.replace("/usr/bin/gnome-shell", "/tmp/gnome-shell"),
                        DESKTOP_ROWS.replace("/usr/bin/gnome-shell", "/usr/bin/gnome-shell --extra"),
                        DESKTOP_ROWS.replace("/usr/bin/gnome-shell", "/usr/bin/python3"),
                        DESKTOP_ROWS + "\n|    0 N/A N/A 999 C /usr/bin/python3 100MiB |"):
            with self.subTest(rows=changed), self.assertRaisesRegex(ValueError, "proceso, ruta o tipo"):
                runtime.gpu_idle_diagnosis(self.record(changed))

    def test_busy_unknown_utilization_failed_queries_and_compute_activity_fail(self):
        for util in (1, 100, "N/A", "desconocida"):
            with self.subTest(util=util), self.assertRaisesRegex(ValueError, "utilización 0"):
                runtime.gpu_idle_diagnosis(self.record(util=util))
        for query in ("gpu_apps", "gpu_summary"):
            for change in ({"returncode": 1}, {"returncode": None}, {"timed_out": True}):
                record = self.record()
                record[query].update(change)
                with self.subTest(query=query, change=change), self.assertRaisesRegex(ValueError, "consulta GPU"):
                    runtime.gpu_idle_diagnosis(record)
        record = self.record()
        record["gpu_apps"]["stdout"] = "9001, /usr/bin/python3\n"
        with self.assertRaisesRegex(ValueError, "cómputo"):
            runtime.gpu_idle_diagnosis(record)

    def test_unreadable_truncated_ambiguous_or_mixed_process_tables_fail(self):
        valid = self.record()["gpu_summary"]["stdout"]
        broken = ("No running processes found", valid.rsplit("+", 2)[0],
                  valid.replace("GPU-Util", "GPU-????"), valid.replace("PID   Type", "PID   ???"),
                  valid.replace(DESKTOP_ROWS, ""), valid.replace(DESKTOP_ROWS, DESKTOP_ROWS + "\n" + EMPTY_PROCESSES),
                  valid.replace(DESKTOP_ROWS, DESKTOP_ROWS + "\n| fila ilegible |"),
                  valid.replace("2868", "3018"), valid + "\nsalida inesperada",
                  valid.replace("|   0  NVIDIA", "|   1  NVIDIA"))
        for text in broken:
            record = self.record()
            record["gpu_summary"]["stdout"] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                runtime.gpu_idle_diagnosis(record)

    def host(self):
        config = protocol.load_config()
        node = protocol.cluster_config(config)["nodes"][0]
        network = json.dumps([{"ifname": "enp1s0f0np0", "flags": ["LOWER_UP"],
                               "addr_info": [{"local": config["network"]["node_ips"][node["node_id"]]}]}])
        image = {"returncode": 0, "stdout": json.dumps(config["image"]["id"]) + " " +
                 json.dumps(config["image"]["architecture"])}
        snapshot = {"complete": True, "metadata_sha256": {"config.json": "b" * 64},
                    "shards": [{"name": "weights.safetensors", "blob_name": "a" * 64, "size_bytes": 16}]}
        host = {**self.record(), "hostname": node["expected_hostname"], "uid": 1000,
                "gpu": {"returncode": 0, "stdout": "GPU-fixture, NVIDIA GB10, 580.173.02\n"},
                "image": image, "docker_running": {"returncode": 0, "stdout": ""},
                "network": {"returncode": 0, "stdout": network}, "rdma_devices": ["rdma_cm", "uverbs0"],
                "ports_in_use": [], "models": {config["model"]["key"]: snapshot},
                "weight_sha256": {"weights.safetensors": "a" * 64},
                "memory_bytes": {"MemAvailable": 100 * runtime.GIB},
                "resources": {"cpu_affinity_count": 8, "workspace_writable": True,
                              "checkpoint_disk_free_bytes": 100 * runtime.GIB, "checkpoint_disk_free_inodes": 10_000},
                "sft_container_probe": {"memory_max": "max", "cpu_max": "max 100000", "cpu_affinity_count": 8}}
        return config, node, host

    def test_host_check_retains_original_evidence_and_same_gpu_diagnosis(self):
        config, node, host = self.host()
        summary = host["gpu_summary"]["stdout"]
        runtime._check_host(node, config, host, host)
        self.assertEqual(host["gpu_summary"]["stdout"], summary)
        self.assertEqual(host["gpu_idle_assessment"], runtime.gpu_idle_diagnosis(host))
        for ports in ([29500], None, ""):
            with self.subTest(ports=ports), self.assertRaisesRegex(ValueError, "puerto"):
                runtime._check_host(node, config, host, {**host, "ports_in_use": ports})

    def test_ready_each_phase_keeps_gpu_evidence_and_rechecks_resources_and_ports(self):
        config, node, host = self.host()
        job = next(job for job in protocol.make_plan(config, "pilot")["jobs"] if job["group"] == "single")
        current = {"gpu": host["gpu"]["stdout"], "apps": "", "summary": host["gpu_summary"]["stdout"],
                   "network": host["network"]["stdout"], "ports_in_use": [],
                   "memory_bytes": host["memory_bytes"], "resources": host["resources"]}

        def ready(record, phase="pilot"):
            with patch.object(runtime, "checked", return_value={"stdout": json.dumps(record)}):
                return runtime.check_ready([node], config, {node["node_id"]: host}, 1000,
                                           job=job, phase=phase, workspace=Path("/tmp/sft-fixture"))

        for phase in ("pilot", "resume", "measure"):
            observed = ready(current, phase)[node["node_id"]]
            self.assertEqual(observed["phase"], phase)
            self.assertEqual(observed["gpu_summary"]["stdout"], current["summary"])
            self.assertEqual(observed["gpu_apps"]["stdout"], "")
            self.assertEqual(observed["gpu_idle_assessment"], runtime.gpu_idle_diagnosis(host))
            self.assertEqual(observed["ports_in_use"], [])
            self.assertEqual(observed["assessment"]["effective_unified_available_bytes"], 100 * runtime.GIB)
        for invalid in ({**current, "ports_in_use": [29500]}, {**current, "ports_in_use": None},
                        {**current, "memory_bytes": {"MemAvailable": runtime.GIB}},
                        {**current, "resources": {**current["resources"], "checkpoint_disk_free_bytes": 0}},
                        {**current, "resources": {**current["resources"], "cpu_affinity_count": 0}},
                        {**current, "apps": "42, python3"},
                        {**current, "summary": current["summary"].replace("0%", "1%")},
                        {**current, "summary": current["summary"].replace(" G ", " C+G ")}):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                ready(invalid)
        with patch.object(runtime, "checked", side_effect=RuntimeError("consulta fallida")), self.assertRaises(RuntimeError):
            runtime.check_ready([node], config, {node["node_id"]: host}, 1000,
                                job=job, phase="pilot", workspace=Path("/tmp/sft-fixture"))

    def test_cpu_probe_retains_bounded_logs_exit_code_and_host_user(self):
        config = protocol.load_config()
        probe = {"uid": 1000, "gid": 1000, "groups": [1000], "user": "study"}
        with (patch.object(runtime, "create", return_value="a" * 64) as create,
              patch.object(runtime, "start"), patch.object(runtime, "checked", return_value={"stdout": "17\n"}),
              patch.object(runtime, "logs", return_value={"stdout": "x" * 5000 + "stdout final",
                           "stderr": "y" * 5000 + "ImportError: fixture"}) as logs):
            with self.assertRaisesRegex(RuntimeError, "exit='17'") as caught:
                runtime._cpu_probe({"node_id": "pegasus"}, config, probe, "campaign", "owner", 1000)
            message = str(caught.exception)
            self.assertIn("ImportError: fixture", message)
            self.assertIn("stdout final", message)
            self.assertLess(len(message), 4300)
            logs.assert_called_once()
            command = create.call_args.args[1]
            self.assertIn("USER=study", command)
            self.assertIn("LOGNAME=study", command)

    def test_cpu_probe_keeps_exit_code_when_logs_cannot_be_collected(self):
        probe = {"uid": 1000, "gid": 1000, "groups": [1000], "user": "study"}
        with (patch.object(runtime, "create", return_value="a" * 64), patch.object(runtime, "start"),
              patch.object(runtime, "checked", return_value={"stdout": "23\n"}),
              patch.object(runtime, "logs", side_effect=RuntimeError("sin logs")),
              self.assertRaisesRegex(RuntimeError, "exit='23'.*sin logs")):
            runtime._cpu_probe({"node_id": "pegasus"}, protocol.load_config(), probe, "campaign", "owner", 1000)

    def test_cpu_probe_zero_exit_with_unreadable_json_preserves_both_log_tails(self):
        config = protocol.load_config()
        probe = {"uid": 1000, "gid": 1000, "groups": [1000], "user": "study"}
        for stdout in ("", " \n", "sin JSON", "x" * 5000 + "stdout final sin JSON", "[]", "null"):
            with (self.subTest(stdout=stdout), patch.object(runtime, "create", return_value="a" * 64),
                  patch.object(runtime, "start"), patch.object(runtime, "checked", return_value={"stdout": "0\n"}),
                  patch.object(runtime, "logs", return_value={"stdout": stdout,
                               "stderr": "y" * 5000 + "stderr final conservado"}),
                  self.assertRaisesRegex(RuntimeError, "sin objeto JSON interpretable.*exit='0'") as caught):
                runtime._cpu_probe({"node_id": "pegasus"}, config, probe, "campaign", "owner", 1000)
            message = str(caught.exception)
            self.assertIn(f"stdout={stdout[-2048:]!r}", message)
            self.assertIn("stderr final conservado", message)
            self.assertLess(len(message), 4300)


class SftRunnerTests(unittest.TestCase):
    def _synthetic_inputs(self, root):
        config = protocol.load_config()
        inventory = protocol.cluster_config(config)
        node_records = [
            {"node_id": node["node_id"], "hostname": node["expected_hostname"],
             "architecture": "aarch64", "uid": 1000, "gid": 1000, "groups": [1000], "user": "fixture",
             "gpu": {"returncode": 0, "stdout": f"GPU-fixture-{index}, NVIDIA GB10, 580.1\n"},
             "models": {config["model"]["key"]: {"metadata_sha256": {}, "shards": []}}}
            for index, node in enumerate(inventory["nodes"])
        ]
        reference = root / "preflight-fixture.json"
        inference.write_new(reference, {"status": "ready_for_smoke", "errors": [], "nodes": node_records})
        config["reference_preflight"] = reference.name
        config["reference_preflight_sha256"] = manifest.sha256_file(reference)
        source = root / "sft-fixture.json"
        inference.write_new(source, config)
        for name in ("infraestructura", "experimentos"):
            (root / name).symlink_to(runner.REPO_ROOT / name, target_is_directory=True)
        with patch.object(runner, "REPO_ROOT", root):
            paths, files = runner._read_inputs(source, config)
        # Mantener todas las validaciones activas con la huella de esta receta de prueba.
        recipe = {key: config[key] for key in protocol._RECIPE_FIELDS}
        encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.enterContext(patch.object(protocol, "_RECIPE_SHA256", hashlib.sha256(encoded).hexdigest()))
        return config, paths, files

    def test_staged_worker_imports_and_loads_inventory_from_its_own_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _, inputs = self._synthetic_inputs(Path(temporary))
            files = runner._worker_files(inputs)
            workspace = Path(temporary) / "isolated"
            env = {"PATH": os.defpath, "PYTHONNOUSERSITE": "1"}

            def local_stage(_node, argv, _deadline, *, stdin, **_kwargs):
                result = subprocess.run([sys.executable, *argv[1:]], input=stdin, cwd=temporary,
                                        env=env, text=True, capture_output=True, check=True, timeout=3)
                return {"stdout": result.stdout}

            with patch.object(runtime, "checked", side_effect=local_stage):
                receipt = runtime.stage({"node_id": "pegasus"}, workspace, files, runner._hashes(files),
                                        {"uid": os.getuid()}, 1000)
            self.assertIn("code/infraestructura/inference_observation.py", receipt["files"])
            self.assertEqual(receipt["files"][f"code/{config['cluster_config']}"],
                             runner._hashes(inputs)["cluster.json"])
            code = workspace / "code"
            help_result = subprocess.run([sys.executable, "-B", "-m", "infraestructura.sft_train", "--help"],
                cwd=code, env=env, capture_output=True, text=True, check=True, timeout=3)
            self.assertIn("--checkpoint-dir", help_result.stdout)
            script = """
import json
from pathlib import Path
from infraestructura import sft_protocol as protocol
assert protocol.REPO_ROOT == Path.cwd()
inventory = protocol.cluster_config(json.loads(Path('../config.json').read_bytes()))
assert inventory['groups']['four'] == ['pegasus', 'octopus', 'griffin', 'draggon']
"""
            subprocess.run([sys.executable, "-B", "-c", script], cwd=code, env=env,
                           capture_output=True, text=True, check=True, timeout=3)

    def test_checkpoint_inode_budget_accumulates_retained_shards(self):
        config = protocol.load_config()
        jobs = protocol.make_plan(config, "pilot")["jobs"]
        nodes = protocol.cluster_config(config)["nodes"]
        limits = runtime.resource_limits(config, nodes, jobs)
        self.assertGreater(limits["pegasus"]["checkpoint_inodes"], 3 * 293)
        self.assertGreater(limits["griffin"]["checkpoint_inodes"], 293)
        self.assertGreater(limits["pegasus"]["checkpoint_bytes"], limits["griffin"]["checkpoint_bytes"])

    def test_preflight_admits_four_with_40_gib_and_retains_single_node_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, files = self._synthetic_inputs(root)
            inventory = protocol.cluster_config(config)
            jobs = protocol.make_plan(config, "pilot")["jobs"]
            historical = json.loads(files["reference/preflight.json"])
            by_node = {row["node_id"]: row for row in historical["nodes"]}

            def host_probe(node, *_args):
                name = node["node_id"]
                return {**by_node[name], "memory_bytes": {"MemAvailable": (40 if name == "pegasus" else 100) * runtime.GIB},
                        "resources": {"cpu_affinity_count": 8, "workspace_writable": True,
                                      "checkpoint_disk_free_inodes": 10_000,
                                      "checkpoint_disk_free_bytes": 100 * runtime.GIB}}

            cpu = {"memory_max": "max", "cpu_max": "max 100000", "cpu_affinity_count": 8}
            deadline = runner.time.monotonic() + 60
            with (patch.object(runtime, "REPO_ROOT", root),
                  patch.object(runtime, "_host_probe", side_effect=host_probe),
                  patch.object(runtime, "_check_host"),
                  patch.object(runtime, "_cpu_probe", return_value=cpu),
                  patch.object(runtime, "_check_cpu"),
                  patch.object(runtime, "cleanup", return_value={"removal_confirmed": True})):
                result = runtime.preflight(config, inventory, jobs, "campaign", "owner", deadline, deadline + 300)

            self.assertEqual(result["errors"], [])
            self.assertEqual(result["status"], "ready")
            pegasus = next(row for row in result["nodes"] if row["node_id"] == "pegasus")
            assessment = pegasus["resource_assessment"]
            self.assertEqual(assessment["effective_unified_available_bytes"], 40 * runtime.GIB)
            self.assertLess(assessment["minimum_unified_bytes"], 40 * runtime.GIB)
            single = next(job for job in jobs if job["group"] == "single")
            single_limit = runtime.resource_limits(config, inventory["nodes"], [single])["pegasus"]
            self.assertEqual(pegasus["planned_peak_and_retained_checkpoints"]["minimum_unified_bytes"],
                             single_limit["minimum_unified_bytes"])
            with self.assertRaisesRegex(ValueError, "memoria unificada"):
                runtime._check_resources(pegasus, single_limit)

    def test_matrix_after_publication_requires_identical_executable_inputs(self):
        hashes = {"requested_config": "a" * 64, "cluster.json": "b" * 64, "code/worker.py": "c" * 64}
        pilot = {"campaign_id": "pilot-fixture", "campaign_sha256": "d" * 64,
                 "stage": "pilot", "status": "complete", "errors": [], "removal_confirmed": True,
                 "qualified_groups": ["four", "pair"], "git": {"commit": "e" * 40},
                 "estimated_matrix_seconds": 60, "input_hashes": hashes,
                 "config_sha256": hashes["requested_config"], "cluster_sha256": hashes["cluster.json"]}
        matrix_commit = "f" * 40
        published = json.dumps(pilot, indent=2).encode()
        default_path = f"docs/resultados/sft-piloto-{pilot['campaign_id']}.json"
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(runner.sft_report, "build_report", return_value=pilot),
              patch.object(runner, "_git_read", return_value=published) as git_read):
            root = Path(tmp)
            directory = root / "sft-campaign" / pilot["campaign_id"]
            groups, reference, archived = runner._qualify(directory, root, hashes, 800, 300, git_commit=matrix_commit)
            self.assertEqual(groups, ["four", "pair"])
            self.assertEqual(reference["git_commit"], pilot["git"]["commit"])
            self.assertEqual(reference["published_report"], {"path": default_path,
                "sha256": hashlib.sha256(published).hexdigest(), "git_commit": matrix_commit})
            self.assertEqual(archived, published)
            git_read.assert_called_once_with("show", f"{matrix_commit}:{default_path}")
            with self.assertRaisesRegex(ValueError, "huellas"):
                runner._qualify(directory, root, {**hashes, "code/worker.py": "f" * 64}, 800, 300,
                                git_commit=matrix_commit)
            alternative = runner.REPO_ROOT / "docs/otro-informe.json"
            git_read.return_value = b"{}"
            with self.assertRaisesRegex(ValueError, "difiere del piloto"):
                runner._qualify(directory, root, hashes, 800, 300, git_commit=matrix_commit, pilot_report=alternative)
            git_read.assert_called_with("show", f"{matrix_commit}:docs/otro-informe.json")
            git_read.side_effect = subprocess.CalledProcessError(128, "git show")
            with self.assertRaisesRegex(ValueError, "falta un informe"):
                runner._qualify(directory, root, hashes, 800, 300, git_commit=matrix_commit)

    def test_cpu_preflight_distinguishes_distribution_and_runtime_torch_versions(self):
        config = protocol.load_config()
        image = config["image"]
        historic = {"container_probe": {"stdout": json.dumps({"python": "3.12", "packages": {
            name: image[name + "_version"] for name in ("torch", "transformers", "tokenizers")}})}}
        cpu = {"python": "3.12", "gpus_hidden": True, "torch_cuda_version": "13.0",
               "nccl_available": True, "nccl_version": [2, 29, 0],
               **{f"{name}_version": image[f"{name}_version"]
                  for name in ("torch", "torch_runtime", "transformers", "tokenizers")}}
        runtime._check_cpu(config, historic, {"sft_container_probe": cpu})
        for field, wrong in (("torch_version", image["torch_runtime_version"]),
                             ("torch_runtime_version", image["torch_version"])):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "versión .*torch"):
                runtime._check_cpu(config, historic, {"sft_container_probe": {**cpu, field: wrong}})

    def test_worker_container_pins_identity_and_readonly_inputs(self):
        config = protocol.load_config()
        probe = {"uid": 1000, "gid": 1100, "groups": [44, 1100, 109, 44], "user": "study"}
        job = protocol.make_plan(config, "pilot")["jobs"][0]
        worker = protocol.worker_argv(config, job, 2, "resume", config_path="/work/config.json",
            model_dir=Path("/models") / protocol.model_subdir(config), out="/work/out/resume",
            checkpoint_dir="/work/checkpoint", campaign_id="campaign", run_id="run")
        argv = runtime.phase_command(config, probe, Path("/tmp/fixture"), "campaign", "owner",
                                     "sft-fixture", worker, 200, phase="resume")
        self.assertIn("--pull=never", argv)
        self.assertIn("--uts=host", argv)
        self.assertIn("--network=host", argv)
        self.assertIn("1000:1100", argv)
        self.assertIn("44", argv)
        self.assertIn("109", argv)
        self.assertEqual(argv.count("--group-add"), 2)
        self.assertIn("NCCL_SOCKET_IFNAME==enp1s0f0np0", argv)
        self.assertIn("type=bind,src=/tmp/fixture/checkpoint,dst=/work/checkpoint,readonly", argv)
        self.assertIn("type=bind,src=" + config["paths"]["hf_hub"] + ",dst=/models,readonly", argv)
        self.assertEqual(argv[-len(worker):], worker)
        self.assertEqual(worker[worker.index("--master-port") + 1], "29500")
        self.assertEqual(worker[worker.index("--node-rank") + 1], "2")
        self.assertIn(config["image"]["id"], argv)

    def test_cleanup_never_removes_id_with_conflicting_ownership(self):
        nodes = [{"node_id": "one"}, {"node_id": "two"}]
        own_id = "a" * 64
        calls = []
        listings = {"one": [own_id, ""], "two": [own_id]}

        def command(node, argv, *args, **kwargs):
            host = node["node_id"]
            calls.append((host, argv))
            if argv[1] == "ps":
                return {"stdout": listings[host].pop(0)}
            if argv[1] == "inspect":
                return {"stdout": "owner campaign" if host == "one" else "foreign campaign"}
            return {"stdout": own_id}

        with patch.object(runtime, "checked", side_effect=command):
            receipt = runtime.cleanup(nodes, "campaign", "owner", 1000)
        removed = [(host, argv[-1]) for host, argv in calls if argv[1] == "rm"]
        self.assertEqual(removed, [("one", own_id)])
        self.assertFalse(receipt["removal_confirmed"])
        self.assertTrue(receipt["nodes"]["one"]["confirmed"])
        self.assertFalse(receipt["nodes"]["two"]["confirmed"])

    def test_oom_requires_failure_on_an_exited_rank(self):
        def write_failure(directory, name):
            path = directory / "raw" / "nodes" / name / "pilot" / "failure.json"
            inference.write_new(path, {"type": "cuda_oom"})

        error = {"Status": "exited", "ExitCode": 1, "OOMKilled": False}
        active = {"Status": "running"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_failure(directory, "pegasus")
            self.assertEqual(runner._phase_outcome(directory, "pilot",
                {"pegasus": error, "octopus": active}, ["pegasus", "octopus"]), "oom")
            with self.assertRaises(ValueError):
                runner._phase_outcome(directory, "pilot", {"pegasus": error, "octopus": error},
                                      ["pegasus", "octopus"])

    def test_run_oom_round_trips_through_manifest_and_reader(self):
        fixture_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config, paths, files = self._synthetic_inputs(fixture_root)
        inventory = protocol.cluster_config(config)
        history = json.loads(files["reference/preflight.json"])
        job = protocol.make_plan(config, "pilot")["jobs"][0]
        git = {"commit": "b" * 40, "dirty": False}
        _, roster = runtime._hardware_roster(history)
        nodes = [{**node, "gpu_uuid": roster[node["node_id"]]["gpu_uuid"]} for node in history["nodes"]]
        states = {"Status": "exited", "ExitCode": 1, "OOMKilled": False, "Error": ""}
        ids = {name: f"{index + 1:064x}" for index, name in enumerate(job["nodes"])}
        cleanup_calls = []
        omit_group = False

        def cleanup(all_nodes, *_args):
            cleanup_calls.append([node["node_id"] for node in all_nodes])
            return {"removal_confirmed": True, "ended_at": manifest.utc_now(), "nodes": {
                name: {"confirmed": True, "remaining_ids": [],
                       "removed_ids": [cid] if len(cleanup_calls) == 1 else []}
                for name, cid in ids.items()}}

        def phase_identities(workspace, phase):
            return [{"node_id": name, "rank": rank,
                "hostname": roster[name]["hostname"], "gpu_uuid": roster[name]["gpu_uuid"],
                "world_size": len(nodes), "phase": phase, "campaign_id": workspace.parent.name,
                "run_id": workspace.name} for rank, name in enumerate(job["nodes"])]

        def collect(node, workspace, phase, *_args):
            name = node["node_id"]
            participants = phase_identities(workspace, phase)
            identity = participants[job["nodes"].index(name)]
            data = {"identity.json": json.dumps(identity).encode()}
            if not (omit_group and name == "octopus"):
                data["group.json"] = json.dumps({**identity, "backend": "nccl", "participants": participants}).encode()
            if name == "pegasus":
                data["failure.json"] = json.dumps({**identity, "type": "cuda_oom", "status": "cuda_oom",
                    "exception_type": "torch.OutOfMemoryError", "oom_origin": "cuda",
                    "message": "CUDA out of memory. Tried to allocate 512 MiB.", "stage": "train_step",
                    "timestamp_utc": manifest.utc_now()}).encode()
            return data

        def ready(selected, _config, _probes, _deadline, *, job, phase, workspace):
            limits = runtime.resource_limits(config, selected, [job])
            host = {"gpu_summary": {"returncode": 0, "stdout": GB10_SUMMARY.format(util=0, processes=DESKTOP_ROWS)},
                    "gpu_apps": {"returncode": 0, "stdout": ""},
                    "memory_bytes": {"MemAvailable": 100 * runtime.GIB},
                    "resources": {"cpu_affinity_count": 8, "workspace_writable": True,
                                  "workspace": str(workspace), "checkpoint_disk_free_inodes": 10_000,
                                  "checkpoint_disk_free_bytes": 100 * runtime.GIB},
                    "sft_container_probe": {"memory_max": "max", "cpu_max": "max 100000", "cpu_affinity_count": 8}}
            return {name: {"node_id": name, "phase": phase, "observed_at": manifest.utc_now(),
                           "memory_bytes": host["memory_bytes"], "resources": host["resources"],
                           "gpu_summary": host["gpu_summary"], "gpu_apps": host["gpu_apps"],
                           "gpu_idle_assessment": runtime.gpu_idle_diagnosis(host), "ports_in_use": [],
                           "assessment": runtime._check_resources(host, limits[name])}
                    for name in job["nodes"]}

        def logs(node, cid, _deadline):
            return {"command": ["docker", "logs", "--tail", "5000", cid], "returncode": 0, "timed_out": False,
                    "stdout": "NCCL INFO Init COMPLETE\n", "stderr": "CUDA out of memory\n"
                    if node["node_id"] == "pegasus" else "", "duration_seconds": 0.01}

        with (tempfile.TemporaryDirectory() as temporary,
              patch.object(manifest, "capture_git", return_value=git),
              patch.object(manifest, "capture_system", return_value={}),
              patch.object(manifest, "capture_packages", return_value={}),
              patch.object(manifest, "capture_nvidia", return_value={}),
              patch.object(runtime, "check_ready", side_effect=ready),
              patch.object(runtime, "stage", side_effect=lambda _n, workspace, _f, hashes, probe, _d:
                           {"workspace": str(workspace), "files": hashes, "uid": probe["uid"]}),
              patch.object(runtime, "create", side_effect=lambda node, *_: ids[node["node_id"]]),
              patch.object(runtime, "start"),
              patch.object(runtime, "state", side_effect=lambda node, *_:
                           states if node["node_id"] == "pegasus" else {**states, "Status": "running"}),
              patch.object(runtime, "logs", side_effect=logs),
              patch.object(runtime, "collect", side_effect=collect),
              patch.object(runtime, "cleanup", side_effect=cleanup),
              patch.object(runner, "_event")):
            root = Path(temporary)
            campaign = root / "sft-campaign" / "campaign-fixture"
            for name, data in files.items():
                path = campaign / ("config.json" if name == "requested_config" else f"inputs/{name}")
                runner._bytes_new(path, data)
            deadline = runner.time.monotonic() + 600
            ref, removed, errors = runner._job(config, inventory, {"nodes": nodes}, campaign, job, campaign.name,
                "fixture", paths, files, git, root, deadline, deadline + 300)
            self.assertTrue(removed)
            self.assertEqual(errors, [])
            run_dir = root / "full-sft" / ref["run_id"]
            self.assertEqual(set(inference.read_json(run_dir / "logs/pilot-resources.json")), set(job["nodes"]))
            self.assertEqual(manifest.load_run(run_dir)["schema_version"], "2.2")
            self.assertEqual((ref["status"], ref["outcome"]), ("failed", "oom"))
            self.assertEqual(runner.sft_report.assess_run(run_dir, job, config)["outcome"], "oom")
            self.assertEqual(cleanup_calls, [job["nodes"], job["nodes"]])
            omit_group = True
            cleanup_calls.clear()
            ref, removed, errors = runner._job(config, inventory, {"nodes": nodes}, campaign, job, campaign.name,
                "fixture", paths, files, git, root, deadline, deadline + 300)
            self.assertTrue(removed)
            self.assertEqual((ref["status"], ref["outcome"]), ("failed", "indeterminate"))
            self.assertTrue(any("group.json" in error for error in errors))
            self.assertEqual(cleanup_calls, [job["nodes"], job["nodes"]])

    def test_matrix_archives_publication_without_changing_scientific_input_hashes(self):
        fixture_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _, paths, files = self._synthetic_inputs(fixture_root)
        hashes = runner._hashes(files)
        git = {"commit": "b" * 40, "dirty": False}
        pilot = {"campaign_id": "pilot-fixture", "campaign_sha256": "d" * 64,
                 "stage": "pilot", "status": "complete", "errors": [], "removal_confirmed": True,
                 "qualified_groups": ["four"], "git": {"commit": "e" * 40},
                 "estimated_matrix_seconds": 60, "input_hashes": hashes,
                 "config_sha256": hashes["requested_config"], "cluster_sha256": hashes["cluster.json"]}
        published = json.dumps(pilot, indent=2).encode() + b"\n"
        root = fixture_root / "results"
        with (patch.object(runner, "_read_inputs", return_value=(paths, files)),
              patch.object(runner, "_verify_tracked"), patch.object(runner, "_event"),
              patch.object(runner, "reservation", return_value=nullcontext()),
              patch.object(manifest, "capture_git", return_value=git),
              patch.object(runner.sft_report, "build_report", return_value=pilot),
              patch.object(runner, "_git_read", return_value=published),
              patch.object(runtime, "preflight", return_value={"errors": ["fixture"], "nodes": [],
                           "removal": {"removal_confirmed": True}}),
              patch.object(runtime, "cleanup", return_value={"removal_confirmed": True})):
            directory = runner.run_campaign(paths["requested_config"], "run", 800,
                root / "sft-campaign" / pilot["campaign_id"], results_root=root, pilot_report="docs/pilot.json")
        campaign = inference.read_json(directory / "campaign.json")
        self.assertEqual(campaign["input_hashes"], hashes)
        self.assertEqual((directory / "published-pilot.json").read_bytes(), published)
        self.assertEqual(campaign["pilot_reference"]["published_report"], {"path": "docs/pilot.json",
            "sha256": hashlib.sha256(published).hexdigest(), "git_commit": git["commit"]})

    def test_campaign_continues_after_documented_oom_and_seals_partial_on_unexpected_failure(self):
        fixture_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _, paths, files = self._synthetic_inputs(fixture_root)
        source = paths["requested_config"]
        git = {"commit": "b" * 40, "dirty": False}
        invoked = []

        def job(*args):
            planned = args[4]
            invoked.append(planned["group"])
            ref = {"job": planned["name"], "group": planned["group"], "block": 0, "run_id": planned["name"],
                "status": "failed", "outcome": "oom" if len(invoked) == 1 else "indeterminate",
                "manifest_sha256": "a" * 64, "duration_seconds": 1}
            return ref, True, [] if len(invoked) == 1 else ["fallo inesperado"]

        with (tempfile.TemporaryDirectory() as temporary,
              patch.object(runner, "_read_inputs", return_value=(paths, files)),
              patch.object(runner, "_verify_tracked"),
              patch.object(runner, "_event"),
              patch.object(runner, "reservation", return_value=nullcontext()),
              patch.object(manifest, "capture_git", return_value=git),
              patch.object(runtime, "preflight", return_value={"errors": [], "nodes": [],
                           "removal": {"removal_confirmed": True}}),
              patch.object(runtime, "cleanup", return_value={"removal_confirmed": True}),
              patch.object(runner, "_job", side_effect=job)):
            output = runner.run_campaign(source, "pilot", 720, results_root=temporary)
            self.assertEqual(invoked, ["four", "pair"])
            self.assertFalse((output / "COMPLETE").exists())
            marker = json.loads((output / "PARTIAL").read_bytes())
            expected_hashes = {f"{name}_sha256": manifest.sha256_file(output / f"{name}.json")
                               for name in runner.CAMPAIGN_FILES}
            self.assertEqual({key: marker[key] for key in expected_hashes}, expected_hashes)
            campaign = json.loads((output / "campaign.json").read_bytes())
            self.assertEqual([row["outcome"] for row in campaign["runs"]], ["oom", "indeterminate"])
            self.assertTrue(campaign["removal_confirmed"])

    def test_campaign_seal_accounts_for_late_hash_serialization_and_clock_skew(self):
        origin = datetime(2026, 11, 22, tzinfo=timezone.utc)
        for delay in ("none", "under_limit", "hash", "marker", "utc"):
            with self.subTest(delay=delay), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "campaign"
                directory.mkdir()
                campaign = {"campaign_id": directory.name, "started_at": origin.isoformat(),
                            "limit_seconds": 10, "errors": []}
                elapsed, skew = 2, 0
                for name in runner.CAMPAIGN_FILES:
                    if name != "campaign":
                        inference.write_new(directory / f"{name}.json", {"source": name})
                digest = manifest.sha256_file
                write = inference.write_new

                def hash_prepared(path):
                    nonlocal elapsed
                    if delay == "hash" and path.name == "campaign.json":
                        elapsed = 11
                    return digest(path)

                def write_prepared(path, data):
                    nonlocal elapsed, skew
                    write(path, data)
                    if path.name == "marker.json":
                        if delay == "under_limit":
                            elapsed = 7
                        if delay == "marker":
                            elapsed = 12
                        if delay == "utc":
                            skew = 3

                def wall_clock():
                    return (origin + timedelta(seconds=elapsed + skew)).isoformat()

                with (patch.object(runner.time, "monotonic", side_effect=lambda: 100 + elapsed),
                      patch.object(manifest, "utc_now", side_effect=wall_clock),
                      patch.object(manifest, "sha256_file", side_effect=hash_prepared),
                      patch.object(inference, "write_new", side_effect=write_prepared)):
                    runner._seal_campaign(directory, campaign, 100, complete=True)
                complete = delay in ("none", "under_limit")
                marker_name = "COMPLETE" if complete else "PARTIAL"
                marker = inference.read_json(directory / marker_name)
                index_path = directory / "campaign.json"
                index = inference.read_json(index_path)
                self.assertFalse((directory / ("PARTIAL" if complete else "COMPLETE")).exists())
                self.assertEqual(marker["duration_seconds"], elapsed)
                self.assertEqual(marker["sealed_at"], wall_clock())
                self.assertEqual(marker["campaign_sha256"], digest(index_path))
                self.assertEqual(marker["cleanup_sha256"], digest(directory / "cleanup.json"))
                receipt = inference.read_json(directory / "publication.json")
                self.assertEqual(receipt, {"campaign_id": campaign["campaign_id"], "status": marker_name.lower(),
                    "marker_sha256": digest(directory / marker_name), "observed_at": wall_clock(),
                    "duration_seconds": elapsed})
                if not complete:
                    expected = "discrepancia" if delay == "utc" else "presupuesto"
                    self.assertTrue(any(expected in error for error in index["errors"]), index["errors"])
                with self.assertRaises(FileExistsError):
                    runner._seal_campaign(directory, campaign, 100, complete=True)
                self.assertEqual(inference.read_json(directory / marker_name), marker)

    def test_slow_marker_link_exceeds_both_session_limits_before_confirmation(self):
        origin = datetime(2026, 11, 22, tzinfo=timezone.utc)
        for stage, limit in (("pilot", 7200), ("run", 28800)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "campaign"
                directory.mkdir()
                campaign = {"campaign_id": directory.name, "stage": stage, "started_at": origin.isoformat(),
                            "limit_seconds": limit, "errors": []}
                for name in runner.CAMPAIGN_FILES:
                    if name != "campaign":
                        inference.write_new(directory / f"{name}.json", {"source": name})
                elapsed = limit - 0.2
                attempts = []
                link = os.link

                def delayed_link(source, destination):
                    nonlocal elapsed
                    link(source, destination)
                    if destination.parent == directory and destination.name in ("COMPLETE", "PARTIAL"):
                        self.assertFalse((directory / "publication.json").exists())
                        attempts.append(destination.name)
                        if destination.name == "COMPLETE":
                            elapsed += 2

                def wall_clock():
                    return (origin + timedelta(seconds=elapsed)).isoformat()

                with (patch.object(runner.time, "monotonic", side_effect=lambda: 100 + elapsed),
                      patch.object(manifest, "utc_now", side_effect=wall_clock),
                      patch.object(runner.os, "link", side_effect=delayed_link)):
                    runner._seal_campaign(directory, campaign, 100, complete=True)
                self.assertEqual(attempts, ["COMPLETE", "PARTIAL"])
                self.assertFalse((directory / "COMPLETE").exists())
                index = inference.read_json(directory / "campaign.json")
                self.assertEqual(index["status"], "partial")
                self.assertTrue(any("presupuesto" in error for error in index["errors"]))
                receipt = inference.read_json(directory / "publication.json")
                self.assertEqual(receipt["campaign_id"], directory.name)
                self.assertEqual(receipt["status"], "partial")
                self.assertEqual(receipt["marker_sha256"], manifest.sha256_file(directory / "PARTIAL"))
                self.assertAlmostEqual(receipt["duration_seconds"], limit + 1.8, places=5)
                self.assertEqual(receipt["observed_at"], wall_clock())
                self.assertGreater((datetime.fromisoformat(receipt["observed_at"]) - origin).total_seconds(), limit)

    def test_marker_without_publication_receipt_is_left_unconfirmed_on_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "campaign"
            directory.mkdir()
            campaign = {"campaign_id": directory.name, "started_at": manifest.utc_now(),
                        "limit_seconds": 7200, "errors": []}
            for name in runner.CAMPAIGN_FILES:
                if name != "campaign":
                    inference.write_new(directory / f"{name}.json", {})
            write = inference.write_new

            def fail_receipt(path, data):
                if path.name == "publication.json":
                    self.assertTrue((directory / "COMPLETE").is_file())
                    raise OSError("fixture: recibo no publicado")
                return write(path, data)

            with (patch.object(inference, "write_new", side_effect=fail_receipt),
                  self.assertRaisesRegex(OSError, "recibo no publicado")):
                runner._seal_campaign(directory, campaign, runner.time.monotonic(), complete=True)
            self.assertFalse((directory / "publication.json").exists())
