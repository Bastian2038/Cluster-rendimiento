"""CPU-only tests for inference observation and identity validation."""

import unittest
from unittest.mock import patch

from infraestructura import inference_observation as observation


class ObservationTests(unittest.TestCase):
    def test_sample_preserves_na_and_reads_host_counters(self):
        files = {
            "/proc/meminfo": "MemTotal: 1000 kB\nMemAvailable: 400 kB\n",
            "/proc/pressure/memory": "some avg10=0.01 avg60=0.00 avg300=0.00 total=11\n",
            "/proc/stat": "cpu  1 2 3 4 5 6 7 8\n",
            "/sys/class/net/enp1s0f0np0/statistics/rx_bytes": "100\n",
            "/sys/class/net/enp1s0f0np0/statistics/tx_bytes": "200\n",
            "/sys/class/net/enP2p1s0f0np0/statistics/rx_bytes": "300\n",
            "/sys/class/net/enP2p1s0f0np0/statistics/tx_bytes": "400\n",
        }
        gpu = "GPU-abc, N/A, 300, 1000, N/A, 45, 1000, N/A\n"
        apps = "GPU-abc, 123, N/A\n"
        with patch.object(observation, "_read", side_effect=lambda path: files.get(path)), \
             patch.object(observation, "_query", side_effect=[gpu, apps]), \
             patch.object(observation.socket, "gethostname", return_value="pegasus"):
            sample = observation.collect_sample()
        self.assertEqual(sample["memory"]["MemTotal_kib"], 1000)
        self.assertEqual(sample["memory_pressure"]["some"]["total"], 11)
        self.assertEqual(sample["cpu_ticks"]["idle"], 4)
        self.assertEqual(sample["nic_f0"]["enp1s0f0np0"]["tx_bytes"], 200)
        self.assertEqual(sample["nic_f0"]["enP2p1s0f0np0"]["tx_bytes"], 400)
        self.assertIsNone(sample["gpus"][0]["utilization_gpu"])
        self.assertIsNone(sample["gpus"][0]["power_draw"])
        self.assertIsNone(sample["gpu_apps"][0]["used_gpu_memory_mib"])

    def test_missing_sources_remain_missing(self):
        with patch.object(observation, "_read", return_value=None), \
             patch.object(observation, "_query", return_value=None):
            sample = observation.collect_sample()
        self.assertIsNone(sample["memory"]["MemAvailable_kib"])
        self.assertIsNone(sample["nic_f0"]["enp1s0f0np0"]["rx_bytes"])
        self.assertIsNone(sample["nic_f0"]["enP2p1s0f0np0"]["rx_bytes"])
        self.assertEqual(sample["gpus"], [])

    def test_validate_loaded_workers(self):
        records = [
            {"stage": "init_device", "rank": 0},
            {"stage": "load_model", "node_id": "Octopus", "hostname": "octopus",
             "gpu_uuid": "GPU-2", "rank": 1, "world_size": 2,
             "effective_tp": 2, "effective_pp": 1, "model_path": "/m",
             "tokenizer_path": "/m", "dtype": "torch.bfloat16",
             "versions": {"ray": "2", "vllm": "0.24", "torch": "2"}},
            {"stage": "load_model", "node_id": "Pegasus", "hostname": "pegasus",
             "gpu_uuid": "GPU-1", "rank": 0, "world_size": 2,
             "effective_tp": 2, "effective_pp": 1, "model_path": "/m",
             "tokenizer_path": "/m", "dtype": "torch.bfloat16",
             "versions": {"ray": "2", "vllm": "0.24", "torch": "2"}},
        ]
        expected = {"Pegasus": {"expected_hostname": "pegasus"},
                    "Octopus": {"expected_hostname": "octopus"}}
        loaded = observation.validate_worker_records(records, expected, 2, "/m")
        self.assertEqual([item["rank"] for item in loaded], [0, 1])
        records[1]["gpu_uuid"] = None
        with self.assertRaisesRegex(ValueError, "GPU UUID"):
            observation.validate_worker_records(records, expected, 2, "/m")
        records[1]["gpu_uuid"] = "GPU-2"
        for key, bad, error in (
            ("world_size", True, "world size"),
            ("effective_tp", True, "paralelismo"),
            ("tokenizer_path", "/other", "tokenizer"),
            ("dtype", "torch.float16", "dtype"),
            ("versions", {}, "versiones"),
        ):
            with self.subTest(key=key):
                old = records[1][key]
                records[1][key] = bad
                with self.assertRaisesRegex(ValueError, error):
                    observation.validate_worker_records(records, expected, 2, "/m")
                records[1][key] = old


if __name__ == "__main__":
    unittest.main()
