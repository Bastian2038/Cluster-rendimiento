import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from infraestructura import cluster, cluster_inventory, manifest


CONFIG = Path(__file__).resolve().parents[1] / "experimentos/configs/cluster.json"


class ClusterConfigTests(unittest.TestCase):
    def test_load_four_nodes_and_groups(self):
        config = cluster.load_config(CONFIG)
        self.assertEqual([n["node_id"] for n in cluster.selected_nodes(config, "four")],
                         ["pegasus", "octopus", "griffin", "draggon"])
        self.assertEqual(len(cluster.selected_nodes(config, "pair")), 2)
        self.assertEqual(config["switch"]["status"], "reported")

    def test_rejects_injection_duplicates_and_secrets(self):
        original = cluster.load_config(CONFIG)
        for change in (lambda c: c["nodes"][1]["ssh"].update(user="bob;echo"),
                       lambda c: c["nodes"][1].update(node_id="pegasus"),
                       lambda c: c["nodes"][1].update(password="x"),
                       lambda c: c["groups"].update(extra=["pegasus"])):
            config = copy.deepcopy(original)
            change(config)
            with self.assertRaises(ValueError):
                cluster.validate_config(config)

    def test_strict_ssh_command_and_local(self):
        config = cluster.load_config(CONFIG)
        remote = cluster.ssh_command(config["nodes"][1], ["python3", "-c", "print('hi')"])
        self.assertIn("BatchMode=yes", remote)
        self.assertIn("StrictHostKeyChecking=yes", remote)
        self.assertIn("UpdateHostKeys=no", remote)
        self.assertEqual(cluster.ssh_command(config["nodes"][0], ["true"]), ["true"])

    def test_plan_no_network_process_or_writes(self):
        with mock.patch.object(cluster_inventory.subprocess, "run", side_effect=AssertionError("process")), \
             mock.patch.object(cluster_inventory.socket, "gethostname", side_effect=AssertionError("hostname")):
            plan = cluster_inventory.plan(CONFIG)
        self.assertEqual(plan["status"], "planned_only")
        self.assertEqual(len(plan["nodes"]), 4)

    def test_embedded_collector_parses_without_execution(self):
        compile(cluster_inventory.COLLECTOR, "<collector>", "exec")
        self.assertIn("nvidia-smi", cluster_inventory.COLLECTOR)

    def test_switch_reported_and_display_alias_are_declarative(self):
        config = cluster.load_config(CONFIG)
        config["nodes"][0]["alias"] = "Pegasus"
        config["switch"].update(status="reported", manufacturer="reported vendor")
        cluster.validate_config(config)

    def test_lldp_multiple_interfaces_and_missing_status(self):
        payload = {"lldp": {"interface": {
            "eth0": {"chassis": {"switch-a": {"id": {"value": "aa:bb"}, "name": "switch-a"}},
                     "port": {"Ethernet1": {"id": {"value": "Ethernet1"}}}},
            "eth1": {"chassis": {"id": {"value": "cc:dd"}},
                     "port": {"id": {"value": "port2"}}}}}}
        neighbors = cluster_inventory._lldp_neighbors({"status": "ok", "stdout": json.dumps(payload)})
        self.assertEqual([(n["interface"], n["chassis_id"]) for n in neighbors],
                         [("eth0", "aa:bb"), ("eth1", "cc:dd")])
        self.assertEqual(cluster_inventory._lldp_neighbors({"status": "denied"}), [])
        self.assertEqual(cluster_inventory._lldp_neighbors({"status": "missing"}), [])

    def test_lldp_repeated_interface_keeps_all_announcements_and_details(self):
        payload = {"lldp": {"interface": [
            {"eth0": {"chassis": {"MikroTik": {
                "id": {"value": "aa:bb"}, "descr": "RouterOS announced",
                "mgmt-ip": ["192.0.2.1", "fe80::1"]}},
                "port": {"id": {"value": "ether1"}, "descr": "QSFP port"}}},
            {"eth0": {"chassis": {"peer": {
                "id": {"value": "cc:dd"}, "descr": "Peer announced"}},
                "port": {"id": {"value": "eth2"}, "descr": "peer port"}}},
        ]}}
        neighbors = cluster_inventory._lldp_neighbors({"status": "ok", "stdout": json.dumps(payload)})
        self.assertEqual(len(neighbors), 2)
        self.assertEqual([n["interface"] for n in neighbors], ["eth0", "eth0"])
        self.assertEqual([n["chassis_name"] for n in neighbors], ["MikroTik", "peer"])
        self.assertEqual(neighbors[0]["chassis_descr"], "RouterOS announced")
        self.assertEqual(neighbors[0]["mgmt_ips"], ["192.0.2.1", "fe80::1"])
        self.assertEqual(neighbors[0]["port_descr"], "QSFP port")

    def test_nccl_package_filter_handles_historical_literal_newlines(self):
        probe = {"status": "ok", "stdout": "libvncclient1 0.9\\nlibnccl2 2.27\\nnccl-tests 2.17\\n"}
        self.assertEqual(cluster_inventory._nccl_packages(probe),
                         "libnccl2 2.27; nccl-tests 2.17")
        self.assertEqual(cluster_inventory._nccl_packages({"status": "ok", "stdout": "libvncclient1 0.9\n"}),
                         "ningún paquete NCCL registrado por dpkg")
        self.assertIn("'libnccl*','nccl*'", cluster_inventory.COLLECTOR)

    def test_nccl_partial_dpkg_output_retains_valid_package_and_marks_failure(self):
        probe = {"status": "error", "returncode": 1,
                 "stdout": "libnccl2 2.27\\nlibvncclient1 0.9\\n",
                 "stderr": "dpkg-query: no packages found matching nccl*"}
        self.assertEqual(cluster_inventory._nccl_packages(probe),
                         "consulta parcial (dpkg no completó): libnccl2 2.27")
        self.assertEqual(cluster_inventory._nccl_packages({"status": "error", "returncode": 1,
                                                              "stdout": ""}),
                         "consulta dpkg fallida; paquetes NCCL desconocidos")

    def test_missing_dmi_is_unknown_without_false_capture_failure(self):
        nodes = cluster.load_config(CONFIG)["nodes"]
        node = nodes[0]
        peers = {ip for other in nodes[1:] for ip in other["data_addresses"]}
        record = {"hostname": node["expected_hostname"], "rank": None,
                  "probes": {"cpu": {"status": "ok"}, "memory": {"status": "ok"},
                             "os_release": {"status": "ok"}, "interfaces": {"status": "ok"},
                             "addresses": {"status": "ok"},
                             "gpu": {"status": "ok", "stdout": "GPU,GPU-fixture,1,driver\n"},
                             "dmi": {"sys_vendor": {"status": "missing"}},
                             "routes": {ip: {"status": "ok"} for ip in peers}}}
        self.assertEqual(cluster_inventory._required_errors(record, node, nodes), [])


class ClusterLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nodes = cluster.load_config(CONFIG)["nodes"]

    def record(self, node, *, wrong_hostname=False):
        okay = {"status": "ok", "value": "fixture"}
        return {"schema_version": "1.0", "node_id": node["node_id"],
                "hostname": "wrong-host" if wrong_hostname else node["expected_hostname"],
                "rank": None, "probes": {
                    "cpu": okay, "memory": okay, "os_release": okay,
                    "interfaces": {"status": "ok", "value": {"eth0": {}}},
                    "addresses": {"status": "ok", "stdout": "[]"},
                    "gpu": {"status": "ok", "stdout":
                            f"Fixture GPU,GPU-{node['node_id']},1,fixture-driver\n"},
                    "dmi": {"sys_vendor": okay, "product_name": okay},
                    "lldp": {"status": "ok", "stdout": json.dumps({"lldp": {"interface": {
                        "eth0": {"chassis": {"id": {"value": "aa:bb"}, "name": "fixture-switch"},
                                 "port": {"id": {"value": "Ethernet1"}}}}}})},
                    "routes": {address: {"status": "ok", "stdout": "[]"}
                               for peer in self.nodes if peer is not node
                               for address in peer["data_addresses"]},
                }}

    def capture(self, *, wrong_hostname=False, timeout_node=None, malformed_node=None):
        seen = []
        def fake_run(command, **kwargs):
            node = self.nodes[len(seen)]
            seen.append(command)
            if node["node_id"] == timeout_node:
                import subprocess
                raise subprocess.TimeoutExpired(command, 45, output=b'{"event":"probe"}\n')
            if node["node_id"] == malformed_node:
                return SimpleNamespace(returncode=0, stderr="", stdout="[]\n")
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
                self.record(node, wrong_hostname=wrong_hostname and len(seen) == 2)))
        with mock.patch.object(manifest, "capture_git", return_value={
                    "commit": "a" * 40, "dirty": False, "root": str(self.root),
                    "branch": "fixture", "changed_paths": [], "error": None}), \
             mock.patch.object(manifest, "capture_system", return_value={
                    "hostname": "coordinator", "machine": "fixture"}), \
             mock.patch.object(manifest, "capture_nvidia", return_value={}), \
             mock.patch.object(cluster_inventory.subprocess, "run", side_effect=fake_run):
            directory = cluster_inventory.run(CONFIG, results_root=self.root / "raw")
        self.assertEqual(len(seen), 4)
        self.assertEqual(seen[0][:4], ["timeout", "-k", "2s", "42s"])
        self.assertTrue(all("BatchMode=yes" in command for command in seen[1:]))
        return directory

    def test_four_node_fixture_complete_and_report_integrity(self):
        directory = self.capture()
        loaded = manifest.load_run(directory)
        self.assertEqual(loaded["status"], "complete")
        self.assertEqual(len(loaded["participants"]), 4)
        self.assertEqual(loaded["hosts"], [n["expected_hostname"] for n in self.nodes])
        with self.assertRaisesRegex(ValueError, "fuera de raw"):
            cluster_inventory.render(directory, self.root / "raw" / "report.md")
        output = self.root / "report.md"
        cluster_inventory.render(directory, output)
        report = output.read_text()
        self.assertIn("flowchart LR", report)
        self.assertIn("rutas observadas; conectividad pendiente", report)
        self.assertIn("fixture-switch", report)
        self.assertIn("Vecino LLDP", report)
        (directory / "raw/nodes/pegasus.json").write_text("{}")
        with self.assertRaises(manifest.ManifestError):
            cluster_inventory.render(directory, self.root / "corrupt.md")

    def test_report_keeps_lldp_announcements_and_unknown_speed(self):
        original_record = self.record
        def record_with_network(node, **kwargs):
            record = original_record(node, **kwargs)
            record["probes"]["interfaces"]["value"] = {"eth0": {
                "speed": {"status": "ok", "value": "-1"},
                "mtu": {"status": "ok", "value": "1500"},
                "operstate": {"status": "ok", "value": "down"}}}
            record["probes"]["nccl_packages"] = {"status": "ok", "stdout": "libvncclient1 0.9\\n"}
            record["probes"]["lldp"]["stdout"] = json.dumps({"lldp": {"interface": [
                {"eth0": {"chassis": {"MikroTik": {"id": {"value": "aa:bb"},
                    "descr": "RouterOS announced", "mgmt-ip": ["192.0.2.1"]}},
                    "port": {"id": {"value": "ether1"}, "descr": "QSFP port"}}},
                {"eth0": {"chassis": {"peer": {"id": {"value": "cc:dd"}}},
                    "port": {"id": {"value": "eth2"}}}},
            ]}})
            return record
        with mock.patch.object(self, "record", side_effect=record_with_network):
            directory = self.capture()
        output = self.root / "network-report.md"
        cluster_inventory.render(directory, output)
        report = output.read_text()
        self.assertIn("eth0 (velocidad desconocida; MTU 1500; estado down)", report)
        self.assertNotIn("-1 Mb/s", report)
        self.assertIn("MikroTik | RouterOS announced | 192.0.2.1 | ether1 | QSFP port", report)
        self.assertIn("peer | desconocido | desconocido | eth2", report)
        self.assertIn("ningún paquete NCCL registrado por dpkg", report)
        self.assertNotIn("libvncclient1", report)

    def test_wrong_hostname_fails_without_false_participant(self):
        directory = self.capture(wrong_hostname=True)
        loaded = manifest.load_run(directory)
        self.assertEqual(loaded["status"], "failed")
        self.assertEqual(len(loaded["participants"]), 3)
        self.assertNotIn("wrong-host", loaded["hosts"])
        self.assertTrue(any("hostname observado inesperado" in e["message"] for e in loaded["errors"]))

    def test_timeout_preserves_partial_output_and_fails(self):
        directory = self.capture(timeout_node="griffin")
        loaded = manifest.load_run(directory)
        self.assertEqual(loaded["status"], "failed")
        self.assertEqual(len(loaded["participants"]), 3)
        log = json.loads((directory / "logs/griffin.json").read_text())
        self.assertEqual(log["status"], "timeout")
        self.assertIn('"event":"probe"', log["stdout_partial"])

    def test_malformed_host_keeps_other_observations_and_closes_failed(self):
        directory = self.capture(malformed_node="octopus")
        loaded = manifest.load_run(directory)
        self.assertEqual(loaded["status"], "failed")
        self.assertEqual(len(loaded["participants"]), 3)
        self.assertTrue(any("objeto JSON" in e["message"] for e in loaded["errors"]))
        log = json.loads((directory / "logs/octopus.json").read_text())
        self.assertEqual(log["stdout"], "[]\n")


if __name__ == "__main__":
    unittest.main()
