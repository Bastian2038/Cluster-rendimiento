import copy
import unittest

from infraestructura.inference_metrics import (paired_speedup,
                                               summarize_bench_result,
                                               validate_bench_result)


def fixture():
    return {
        "backend": "openai", "num_prompts": 32, "completed": 32, "failed": 0,
        "duration": 16.0, "total_input_tokens": 32 * 512,
        "total_output_tokens": 32 * 128, "request_throughput": 2.0,
        "output_throughput": 256.0, "input_lens": [512] * 32,
        "output_lens": [128] * 32, "errors": [None] * 32,
        "ttfts": [0.1] * 32, "itls": [[0.01] * 127 for _ in range(32)],
        "p50_tpot_ms": 10.0, "p95_tpot_ms": 10.0,
        "p50_e2el_ms": 1370.0, "p95_e2el_ms": 1370.0,
    }


class MetricsTests(unittest.TestCase):
    def test_summary_and_pair(self):
        raw = fixture()
        summary = summarize_bench_result(raw, expected_prompt_tokens=512)
        self.assertEqual(summary["t32_seconds"], 16)
        self.assertEqual(summary["ttft"]["p95_ms"], 100)
        self.assertEqual(summary["raw_counts"]["output_tokens"], 4096)
        two = dict(summary, comparison_key="same-model-work-trial")
        four = dict(summary, comparison_key="same-model-work-trial", t32_seconds=8)
        self.assertEqual(paired_speedup(two, four), {"s4_2": 2, "e4_2": 1})

    def test_rejects_partial_or_mismatched(self):
        for change in (
            {"failed": 1}, {"completed": 31}, {"output_lens": [127] + [128] * 31},
            {"input_lens": [511] + [512] * 31}, {"errors": ["error"] + [None] * 31},
            {"itls": []}, {"duration": 0},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_bench_result(dict(fixture(), **change), expected_prompt_tokens=512)
        summary = summarize_bench_result(fixture(), expected_prompt_tokens=512)
        with self.assertRaises(ValueError):
            paired_speedup(dict(summary, comparison_key="a"), dict(summary, comparison_key="b"))


if __name__ == "__main__":
    unittest.main()
