import unittest

from infraestructura.inference_metrics import (paired_speedup,
                                               summarize_bench_result,
                                               validate_bench_result,
                                               validate_client_identity)
from infraestructura.inference_runtime import model_path


def fixture(*, input_tokens=512, count=32, duration=16.0, model=None, concurrency=1):
    raw = {
        "backend": "openai", "num_prompts": count, "completed": count, "failed": 0,
        "duration": duration, "total_input_tokens": count * input_tokens,
        "total_output_tokens": count * 128, "request_throughput": count / duration,
        "output_throughput": count * 128 / duration, "input_lens": [input_tokens] * count,
        "output_lens": [128] * count, "errors": [None] * count,
        "ttfts": [0.1] * count, "itls": [[0.01] * 127 for _ in range(count)],
        "p50_tpot_ms": 10.0, "p95_tpot_ms": 10.0,
        "p50_e2el_ms": 1370.0, "p95_e2el_ms": 1370.0,
    }
    if model is not None:
        raw.update(model_id=model["key"], tokenizer_id=model_path(model),
                   max_concurrency=concurrency, request_rate="inf")
    return raw


class MetricsTests(unittest.TestCase):
    def test_client_identity_matches_recorded_vllm_fields(self):
        revision = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
        model = {'id': 'Qwen/Qwen3.5-9B', 'key': 'qwen35-9b', 'revision': revision}
        raw = {'model_id': 'qwen35-9b', 'tokenizer_id': '/models/models--Qwen--Qwen3.5-9B/snapshots/' + revision,
               'request_rate': 'inf', 'max_concurrency': 1}
        validate_client_identity(raw, model=model, concurrency=1)
        for key, wrong in (('model_id', model['id']), ('tokenizer_id', model['id']),
                           ('tokenizer_id', raw['tokenizer_id'][:-1]), ('request_rate', float('inf')),
                           ('request_rate', 1), ('max_concurrency', 4), ('max_concurrency', True),
                           ('max_concurrency', 1.0)):
            with self.subTest(key=key, wrong=wrong), self.assertRaisesRegex(ValueError, key):
                validate_client_identity(dict(raw, **{key: wrong}), model=model, concurrency=1)
        for key in raw:
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, key):
                validate_client_identity({k: v for k, v in raw.items() if k != key}, model=model, concurrency=1)

    def test_speedup_checks_node_counts_on_either_trial_when_present(self):
        base = dict(summarize_bench_result(fixture(), expected_prompt_tokens=512), comparison_key='cell')
        single, four = dict(base, node_count=1), dict(base, node_count=4)
        self.assertEqual(paired_speedup(single, four, baseline_nodes=1, scaled_nodes=4),
                         {'s4_1': 1.0, 'e4_1': 0.25})
        for baseline, scaled in ((dict(single, node_count=True), four), (single, dict(four, node_count=2)),
                                 (dict(single, node_count=None), base), (base, dict(four, node_count=4.0))):
            with self.subTest(baseline=baseline.get('node_count'), scaled=scaled.get('node_count')):
                with self.assertRaisesRegex(ValueError, 'node_count'):
                    paired_speedup(baseline, scaled, baseline_nodes=1, scaled_nodes=4)

    def test_warmup_is_not_a_measured_t32(self):
        warmup = fixture(input_tokens=2048, count=16, duration=8)
        summary = summarize_bench_result(warmup, expected_prompt_tokens=2048, expected_count=16)
        self.assertEqual(summary["raw_counts"]["completed"], 16)
        self.assertIsNone(summary["t32_seconds"])
        with self.assertRaises(ValueError):
            summarize_bench_result(warmup, expected_prompt_tokens=2048)

    def test_summary_and_pair(self):
        raw = fixture()
        summary = summarize_bench_result(raw, expected_prompt_tokens=512)
        self.assertEqual(summary["t32_seconds"], 16)
        self.assertEqual(summary["ttft"]["p95_ms"], 100)
        self.assertEqual(summary["raw_counts"]["output_tokens"], 4096)
        two = dict(summary, comparison_key="same-model-work-trial")
        four = dict(summary, comparison_key="same-model-work-trial", t32_seconds=8)
        self.assertEqual(paired_speedup(two, four), {"s4_2": 2, "e4_2": 1})

    def test_single_baseline_speedup_and_efficiency(self):
        summary = summarize_bench_result(fixture(), expected_prompt_tokens=512)
        single = dict(summary, comparison_key="same-model-work-trial", t32_seconds=48)
        two = dict(summary, comparison_key="same-model-work-trial")
        four = dict(two, t32_seconds=8)
        self.assertEqual(paired_speedup(single, two, baseline_nodes=1, scaled_nodes=2),
                         {"s2_1": 3, "e2_1": 1.5})
        self.assertEqual(paired_speedup(single, four, baseline_nodes=1, scaled_nodes=4),
                         {"s4_1": 6, "e4_1": 1.5})

    def test_single_baseline_rejects_invalid_or_different_work(self):
        summary = summarize_bench_result(fixture(), expected_prompt_tokens=512)
        complete = dict(summary, comparison_key="same-model-work-trial")
        different = dict(complete, raw_counts=dict(complete["raw_counts"], input_tokens=32 * 2048))
        incomplete = dict(complete, raw_counts=dict(complete["raw_counts"], completed=31, failed=1))
        for baseline, scaled in ((complete, different), (incomplete, incomplete)):
            with self.subTest(baseline=baseline["raw_counts"], scaled=scaled["raw_counts"]):
                with self.assertRaises(ValueError):
                    paired_speedup(baseline, scaled, baseline_nodes=1, scaled_nodes=4)

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
