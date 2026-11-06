"""Pruebas CPU del plan, parser y control de procesos NCCL."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infraestructura import nccl


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "experimentos/configs/cluster.json"
PROTOCOL = ROOT / "experimentos/configs/nccl-diagnostic.json"


def fake_outputs(plan, cluster):
    by_id = {node["node_id"]: node for node in cluster["nodes"]}
    outputs = []
    for job in plan["jobs"]:
        lines = [f"[1,0]<stdout>:# Collective test starting: {nccl.OPERATIONS[job['operation']]}",
                 f"[1,0]<stdout>:# nThread 1 nGpus 1 minBytes {job['minimum']} maxBytes {job['maximum']} step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0"]
        for rank, node_id in enumerate(job["group"]):
            host = by_id[node_id]["expected_hostname"]
            lines.extend((f"[1,{rank}]<stderr>:NCCL version 2.27.3",
                          f"[1,{rank}]<stderr>:NCCL INFO Using network IB",
                          f"[1,0]<stdout>:# Rank {rank} Group 0 Pid 123 on {host} device 0 [0000:00:00] GPU",
                          f'[1,{rank}]<stdout>:IPRE_RANK_IDENTITY={{"rank":{rank},"hostname":"{host}","gpu_uuid":"GPU-{rank + 1:04d}"}}',
                          f"[1,{rank}]<stdout>:IPRE_ENV=NCCL_DEBUG=INFO",
                          f"[1,{rank}]<stdout>:IPRE_ENV=NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,GRAPH"))
            for phase in ("before", "after"):
                lines.append(f"[1,{rank}]<stdout>:IPRE_TELEMETRY={phase}|GPU-{rank + 1:04d}, 0, 10, 5")
                lines.append(f"[1,{rank}]<stdout>:IPRE_NIC={phase}|eth0|1|2")
        for size in nccl._expected_sizes(job):
            count = size // 4 if job['operation'] not in ('all_gather', 'reduce_scatter') else ((size // 4 // len(job['group'])) // 4) * 4
            redop = 'none' if job['operation'] == 'all_gather' else 'sum'
            in_wrong = 'N/A' if job['operation'] == 'sendrecv' else '0'
            lines.append(f"[1,0]<stdout>:{size} {count} float {redop} -1 10 1 2 0 10 1 2 {in_wrong}")
        outputs.append({"stdout": "\n".join(lines), "stderr": "", "returncode": 0,
                        "timed_out": False, "duration_seconds": 1.0})
    return outputs


class NcclTests(unittest.TestCase):
    def test_rdma_requires_used_ethernet_hca_in_either_log_stream(self):
        stdout = ("NET/IB : Using [0]rocep1s0f0:1/Ethernet [1]rocep1s0f1:1/Ethernet; OOB eth0:1\n"
                  "NET/IB : Made virtual device [0] name=rocep1s0f0 speed=200 ndevs=1 rail=-1 plane=-1\n"
                  "NET/IB : Made virtual device [1] name=rocep1s0f1 speed=200 ndevs=1 rail=-1 plane=-1\n"
                  "NCCL INFO Using network IB\n"
                  "Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/1\n")
        links = ("IPRE_RDMA_LINK=/sys/class/infiniband/rocep1s0f0/ports/1|Ethernet\n"
                 "IPRE_RDMA_LINK=/sys/class/infiniband/rocep1s0f1/ports/1|Ethernet\n")
        self.assertTrue(nccl._rank_rdma_verified(stdout + links, ""))
        self.assertFalse(nccl._rank_rdma_verified(stdout + links.replace("rocep1s0f1/ports/1|Ethernet", "rocep1s0f1/ports/1|InfiniBand"), ""))
        self.assertFalse(nccl._rank_rdma_verified(stdout.replace("via NET/IB/1", "via NET/Socket/1") + links, ""))
        self.assertFalse(nccl._rank_rdma_verified(stdout.replace("via NET/IB/1", "via NET/IB/2") + links, ""))

    def test_effective_protocol_and_sendrecv_checks(self):
        cluster = json.loads(CLUSTER.read_text())
        plan = nccl.make_plan(CLUSTER, PROTOCOL)
        by_name = {job["name"]: (job, output) for job, output in zip(plan["jobs"], fake_outputs(plan, cluster))}
        job, output = by_name["pair-1-sendrecv"]
        full = output["stdout"]
        result = dict(output, rows=nccl.parse_nccl_table(full), transport=nccl.classify_transport(full),
                      ranks=[{"rank": i, "hostname": cluster["nodes"][i]["expected_hostname"]}
                             for i in range(2)])
        nccl._validate_measurement(job, result, full)
        self.assertIsNone(result["rows"][0]["in_place"]["wrong"])
        for changed, error in ((full.replace(" float sum ", " half sum "), "contradice"),
                               (full.replace(" 2(factor) ", " 4(factor) "), "parámetros"),
                               (full.replace(" 1 2 0 10 1 2 N/A", " 1 2 1 10 1 2 N/A"), "wrong != 0")):
            with self.subTest(error=error):
                mutated = dict(result, rows=nccl.parse_nccl_table(changed))
                with self.assertRaisesRegex(nccl.NcclError, error):
                    nccl._validate_measurement(job, mutated, changed)

    def test_transport_missing_rank_fails(self):
        cluster = json.loads(CLUSTER.read_text())
        plan = nccl.make_plan(CLUSTER, PROTOCOL)
        job, output = plan["jobs"][0], fake_outputs(plan, cluster)[0]
        full = output["stdout"]
        partial = full.replace("[1,3]<stderr>:NCCL INFO Using network IB", "")
        result = dict(output, rows=nccl.parse_nccl_table(partial), transport=nccl.classify_transport(partial),
                      ranks=[{"rank": rank, "hostname": cluster["nodes"][rank]["expected_hostname"]}
                             for rank in range(4)])
        with self.assertRaisesRegex(nccl.NcclError, "cobertura"):
            nccl._validate_measurement(job, result, partial)

    def test_plan_has_no_processes_or_writes(self):
        with mock.patch.object(nccl.subprocess, "Popen", side_effect=AssertionError("spawn")), \
             mock.patch.object(nccl, "_write_new", side_effect=AssertionError("write")):
            plan = nccl.make_plan(CLUSTER, PROTOCOL)
        self.assertEqual(plan["resources_max"], {"nodes": 4, "gpus": 4, "ranks": 4})
        self.assertEqual(plan["jobs"][0]["phase"], "smoke")
        self.assertEqual(plan["jobs"][-1]["phase"], "control")
        self.assertEqual(len(plan["jobs"]), 31)
        self.assertEqual(plan["jobs"][1]["group"], ["pegasus", "octopus", "griffin", "draggon"])
        self.assertEqual(plan["jobs"][4]["operation"], "all_gather")
        self.assertEqual(plan["jobs"][4]["minimum"], 64)
        self.assertIn("-b 64", plan["jobs"][4]["command"][-1])
        self.assertEqual(plan["jobs"][19]["group"], ["pegasus", "griffin"])
        self.assertIn("timeout -k 3s 29s", plan["jobs"][0]["command"][-1])
        self.assertEqual(plan["jobs"][0]["command"][-7:-3], ["timeout", "-k", "5s", "34s"])
        for job in plan["jobs"]:
            command = job["command"]
            self.assertEqual(command[command.index("--launch-agent") + 1], "timeout -k 3s 34s orted")
            self.assertEqual(command[command.index("--timeout") + 1], "35")
            self.assertIn(["--mca", "orte_leave_session_attached", "1"],
                          [command[i:i + 3] for i in range(len(command) - 2)])
            self.assertIn(["--mca", "plm_rsh_no_tree_spawn", "1"],
                          [command[i:i + 3] for i in range(len(command) - 2)])

    def test_parser_and_transport(self):
        table = ("[1,0]<stdout>:# 8 2 float sum -1 12.5 0.1 0.2 1 13.5 0.3 0.4 0\n"
                 "[1,0]<stdout>:# 16 4 float sum -1 20.0 0.5 0.6 0 N/A N/A N/A 0\n")
        rows = nccl.parse_nccl_table(table)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["out_of_place"]["wrong"], 1)
        self.assertIsNone(rows[1]["in_place"]["time_us"])
        self.assertEqual(nccl.classify_transport("Bootstrap : Using eth0\n")["data_transport"], "indeterminate")
        self.assertEqual(nccl.classify_transport("NET/IB : Using mlx5_0\n")["data_transport"], "NET/IB")
        mixed_detection = ("[1,0]<stderr>:NET/Socket : Using eth0\n"
                           "[1,0]<stderr>:NCCL INFO Using network IB\n"
                           "[1,1]<stderr>:NCCL INFO Using network IB\n")
        transport = nccl.classify_transport(mixed_detection)
        self.assertEqual(transport["data_transport"], "NET/IB")
        self.assertEqual(transport["by_rank"]["0"]["data_transport"], "NET/IB")
        self.assertEqual(nccl.classify_transport("NET/IB : No device found\nNET/Socket : Using eth0\n")["data_transport"], "NET/Socket")

    def test_rank_evidence_rejects_partial_and_duplicates(self):
        cluster = json.loads(CLUSTER.read_text())
        job = {"group": ["pegasus", "octopus"]}
        first = '[1,0]<stdout>:IPRE_RANK_IDENTITY={"rank":0,"hostname":"spark-cfea","gpu_uuid":"GPU-aaaa"}'
        second = '[1,1]<stdout>:IPRE_RANK_IDENTITY={"rank":1,"hostname":"spark-60c3","gpu_uuid":"GPU-bbbb"}'
        self.assertEqual(len(nccl._observed_ranks(first + "\n" + second, job, cluster)), 2)
        with self.assertRaisesRegex(nccl.NcclError, "incompleta"):
            nccl._observed_ranks(first, job, cluster)
        with self.assertRaisesRegex(nccl.NcclError, "duplicados"):
            nccl._observed_ranks(first + "\n" + second.replace("GPU-bbbb", "GPU-aaaa"), job, cluster)

    def test_run_preflight_failure_closes_failed_without_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "nccl-test" / "run"
            (run_dir / "raw").mkdir(parents=True)
            with mock.patch.object(nccl.manifest, "start_run", return_value=run_dir), \
                 mock.patch.object(nccl, "_preflight", side_effect=nccl.NcclError("ssh failed")), \
                 mock.patch.object(nccl.manifest, "finalize_run") as finalize, \
                 mock.patch.object(nccl, "_execute", side_effect=AssertionError("job spawn")):
                with self.assertRaisesRegex(nccl.NcclError, "FAILED"):
                    nccl.run(CLUSTER, PROTOCOL)
            self.assertEqual(finalize.call_args.kwargs["status"], "failed")
            self.assertFalse(list((run_dir / "raw").glob("*/result.json")))

    def test_execute_timeout_kills_only_own_group(self):
        fake = mock.Mock()
        fake.pid = 12345
        fake.communicate.side_effect = [nccl.subprocess.TimeoutExpired("test", 1), ("partial", "err")]
        fake.returncode = -15
        with mock.patch.object(nccl.subprocess, "Popen", return_value=fake) as popen, \
             mock.patch.object(nccl.os, "killpg") as killpg:
            result = nccl._execute(["fake"], 1)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["stdout"], "partial")
        self.assertEqual(killpg.call_args.args[0], 12345)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_execute_interrupts_cleanup_own_group(self):
        for interruption in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(interruption=type(interruption).__name__):
                fake = mock.Mock(pid=12345)
                fake.communicate.side_effect = [interruption, ("", "")]
                with mock.patch.object(nccl.subprocess, "Popen", return_value=fake), \
                     mock.patch.object(nccl.os, "killpg") as killpg:
                    with self.assertRaises(type(interruption)):
                        nccl._execute(["fake"], 10)
                killpg.assert_called_once_with(12345, nccl.signal.SIGTERM)

    def test_report_rejects_raw_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "resultados/raw/nccl-test/example"
            raw.mkdir(parents=True)
            with self.assertRaisesRegex(nccl.NcclError, "fuera de raw"):
                nccl.report(raw, raw / "summary.json")
            with self.assertRaisesRegex(nccl.NcclError, "fuera de raw"):
                nccl.report(raw, raw.parent / "sibling-summary.json")

    def test_mocked_full_lifecycle_registers_only_first_four(self):
        cluster = json.loads(CLUSTER.read_text())
        for node in cluster["nodes"]:
            node["access_status"] = "verified"
        plan = nccl.make_plan(CLUSTER, PROTOCOL)
        outputs = fake_outputs(plan, cluster)
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            (run_dir / "raw").mkdir(parents=True)
            with mock.patch("infraestructura.cluster.load_config", return_value=cluster), \
                 mock.patch.object(nccl, "_preflight", return_value=[]), \
                 mock.patch.object(nccl, "_execute", side_effect=outputs), \
                 mock.patch.object(nccl.manifest, "start_run", return_value=run_dir), \
                 mock.patch.object(nccl.manifest, "register_participants") as register, \
                 mock.patch.object(nccl.manifest, "finalize_run") as finalize:
                self.assertEqual(nccl.run(CLUSTER, PROTOCOL), run_dir)
            self.assertEqual(len(register.call_args.args[1]), 4)
            self.assertEqual(register.call_count, 1)
            self.assertEqual(finalize.call_args.kwargs["status"], "complete")
            self.assertEqual(len(list((run_dir / "raw").glob("*/result.json"))), 31)

    def test_real_manifest_cpu_fixture_and_reports(self):
        cluster = json.loads(CLUSTER.read_text())
        plan = nccl.make_plan(CLUSTER, PROTOCOL)
        git = {"commit": "a" * 40, "dirty": False, "root": str(ROOT),
               "branch": "test", "changed_paths": [], "error": None}
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(nccl.manifest, "capture_git", return_value=git), \
             mock.patch.object(nccl.manifest, "capture_system", return_value={"hostname": "coordinator"}), \
             mock.patch.object(nccl.manifest, "capture_nvidia", return_value={"gpus": []}), \
             mock.patch.object(nccl, "_preflight", return_value=[]), \
             mock.patch.object(nccl, "_execute", side_effect=fake_outputs(plan, cluster)):
            run_dir = nccl.run(CLUSTER, PROTOCOL, results_root=Path(temp) / "raw")
            loaded = nccl.manifest.load_run(run_dir)
            self.assertEqual(loaded["status"], "complete")
            self.assertEqual(len(loaded["participants"]), 4)
            nccl.manifest.verify_artifacts(run_dir)
            output_json = Path(temp) / "summary.json"
            output_md = Path(temp) / "summary.md"
            nccl.report(run_dir, output_json)
            nccl.report(run_dir, output_md)
            summary = json.loads(output_json.read_text())
            self.assertFalse(summary["pending_jobs"])
            self.assertTrue(all(job["valid"] for job in summary["jobs"]))
            self.assertTrue(summary["functional_success"])
            self.assertFalse(summary["rdma_success"])
            self.assertFalse(summary["overall_success"])
            self.assertIn("## Mediciones seleccionadas", output_md.read_text())

    def test_incomplete_table_fails_closed(self):
        job = {"name": "x", "group": ["pegasus", "octopus"], "minimum": 8, "maximum": 16}
        result = {"returncode": 0, "timed_out": False, "rows": nccl.parse_nccl_table(
            "8 2 float sum -1 10 1 2 0 10 1 2 0"),
            "transport": {"data_transport": "NET/IB"}}
        with self.assertRaisesRegex(nccl.NcclError, "incompleta"):
            nccl._validate_measurement(job, result, "NCCL version 2.27.3")

    def test_banner_drift_from_preflight_fails(self):
        job = {"name": "x", "group": ["pegasus"], "operation": "all_reduce", "minimum": 8, "maximum": 8}
        result = {"returncode": 0, "timed_out": False,
                  "rows": nccl.parse_nccl_table("8 2 float sum -1 10 1 2 0 10 1 2 0"),
                  "transport": {"data_transport": "indeterminate"},
                  "ranks": [{"rank": 0, "hostname": "spark-cfea", "node_id": "pegasus"}]}
        output = ("[1,0]<stdout>:# Collective test starting: all_reduce_perf\n"
                  "[1,0]<stdout>:# nThread 1 nGpus 1 minBytes 8 maxBytes 8 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0\n"
                  "[1,0]<stderr>:NCCL version 2.27.3\n"
                  "[1,0]<stdout>:# Rank 0 Group 0 Pid 123 on spark-cfea device 0 [0] GPU")
        with self.assertRaisesRegex(nccl.NcclError, "difiere"):
            nccl._validate_measurement(job, result, output,
                                       [{"node_id": "pegasus", "nccl_library_versions": [22600]}])

    def test_preflight_records_partial_and_rejects_missing_returncode(self):
        cluster = json.loads(CLUSTER.read_text())
        config = json.loads(PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(nccl, "_execute", return_value={"returncode": None,
                 "stdout": "", "stderr": "missing", "timed_out": False, "duration_seconds": 0}):
            with self.assertRaisesRegex(nccl.NcclError, "preflight falló"):
                nccl._preflight(cluster, config, nccl.time.monotonic() + 30, Path(temp))
            records = list(Path(temp).glob("*.json"))
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0].read_text())["check"]["returncode"], None)

    def test_preflight_compares_binaries_and_mpi_before_gpu_jobs(self):
        cluster = json.loads(CLUSTER.read_text())
        config = json.loads(PROTOCOL.read_text())
        by_id = {node["node_id"]: node for node in cluster["nodes"]}
        for drift, message in ((None, None), ("binary", "ejecutables"),
                               ("mpi", "versión MPI"), ("symbols", "inicialización MPI")):
            with self.subTest(drift=drift):
                def execute(argv, seconds):
                    node_id, cmd = argv[1], argv[6:]
                    if cmd[:1] == ["env"]:
                        cmd = cmd[2:]
                    first = cmd[0]
                    stdout = ""
                    if first == "hostname":
                        stdout = by_id[node_id]["expected_hostname"]
                    elif first == "nvidia-smi":
                        if cmd[1] == "--query-gpu=uuid":
                            stdout = f"GPU-{list(by_id).index(node_id) + 1:04d}"
                        elif cmd[1] == "--query-gpu=memory.free":
                            stdout = "[N/A]"
                        elif cmd[1] == "--query-gpu=driver_version":
                            stdout = "580.fixture"
                    elif first == "cat":
                        stdout = "MemAvailable: 1234 kB\nSwapFree: 0 kB\n"
                    elif first == "mpirun":
                        stdout = "mpirun (Open MPI) " + ("5.0" if drift == "mpi" and node_id == "griffin" else "4.1")
                    elif first == "sha256sum":
                        changed = drift == "binary" and node_id == "griffin" and "_perf" in cmd[1]
                        stdout = ("b" if changed else "a") * 64 + "  " + cmd[1]
                    elif first == "ldd":
                        stdout = "libmpi.so.40 => /lib/libmpi.so.40\nlibnccl.so.2 => /lib/libnccl.so.2\n"
                    elif first == "python3":
                        stdout = "22703\n"
                    elif first == "readelf":
                        stdout = "fixture ELF" if drift == "symbols" else "GLOBAL UND MPI_Init_thread"
                    return {"returncode": 0, "stdout": stdout, "stderr": "",
                            "timed_out": False, "duration_seconds": 0}
                with mock.patch("infraestructura.cluster.ssh_command",
                                side_effect=lambda node, cmd: ["fixture", node["node_id"], *cmd]), \
                     mock.patch.object(nccl, "_execute", side_effect=execute):
                    if message:
                        with self.assertRaisesRegex(nccl.NcclError, message):
                            nccl._preflight(cluster, config, nccl.time.monotonic() + 10)
                    else:
                        evidence = nccl._preflight(cluster, config, nccl.time.monotonic() + 10)
                        self.assertEqual(len(evidence), 4)
                        self.assertEqual(set(evidence[0]["binary_hashes"]), set(nccl.OPERATIONS.values()))

    def test_cancelled_job_keeps_partial_log_and_failed_manifest(self):
        git = {"commit": "a" * 40, "dirty": False, "root": str(ROOT),
               "branch": "test", "changed_paths": [], "error": None}
        interrupted = KeyboardInterrupt()
        interrupted.execution_result = {"returncode": -15, "stdout": "partial rank output\n",
                                        "stderr": "cancelled\n", "timed_out": True,
                                        "interrupted": "KeyboardInterrupt", "duration_seconds": 0.1}
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(nccl.manifest, "capture_git", return_value=git), \
             mock.patch.object(nccl.manifest, "capture_system", return_value={"hostname": "coordinator"}), \
             mock.patch.object(nccl.manifest, "capture_nvidia", return_value={"gpus": []}), \
             mock.patch.object(nccl, "_preflight", return_value=[]), \
             mock.patch.object(nccl, "_execute", side_effect=interrupted):
            with self.assertRaisesRegex(nccl.NcclError, "FAILED"):
                nccl.run(CLUSTER, PROTOCOL, results_root=Path(temp) / "raw")
            run = next((Path(temp) / "raw/nccl-test").iterdir())
            loaded = nccl.manifest.load_run(run)
            self.assertEqual(loaded["status"], "failed")
            self.assertEqual(loaded["participants"], [])
            self.assertEqual((run / "raw/smoke-four/stdout.log").read_text(), "partial rank output\n")
            nccl.manifest.verify_artifacts(run)

    def test_report_excludes_invalid_job_from_dispersion(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "raw" / "nccl-test" / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text("{}")
            planned = []
            for name, value, invalid in (("good", 2, False), ("bad", 999, True)):
                job = {"name": name, "phase": "main", "group": ["pegasus", "octopus"],
                       "operation": "all_reduce"}
                planned.append(job)
                item = {"job": job, "transport": {"data_transport": "NET/IB"},
                        "rows": [{"bytes": 1048576,
                                  "in_place": {"time_us": 10, "algbw_GBps": 1, "busbw_GBps": value},
                                  "out_of_place": {"time_us": 10, "algbw_GBps": 1, "busbw_GBps": value}}],
                        "returncode": 0, "timed_out": False, "ranks": []}
                if invalid:
                    item["validation_error"] = "wrong != 0"
                directory = run_dir / "raw" / name
                directory.mkdir(parents=True)
                (directory / "result.json").write_text(json.dumps(item))
            fake_run = {"run_id": "run", "kind": "nccl-test", "status": "failed",
                        "requested_config": {"jobs": planned}, "errors": [{"message": "wrong"}]}
            output = Path(temp) / "summary.json"
            with mock.patch.object(nccl.manifest, "load_run", return_value=fake_run), \
                 mock.patch.object(nccl.manifest, "verify_artifacts"):
                nccl.report(run_dir, output)
            summary = json.loads(output.read_text())
            self.assertFalse(summary["jobs"][0 if summary["jobs"][0]["name"] == "bad" else 1]["valid"])
            self.assertTrue(all(point["max"] != 999 for point in summary["dispersion"]))


if __name__ == "__main__":
    unittest.main()
