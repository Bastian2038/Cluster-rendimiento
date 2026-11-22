import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from infraestructura.serving_capacity_metrics import capacity_frontiers, summarize_phase


MODEL = "qwen35-9b"
CONFIG = {
    "capacity": {
        "rates_rps": [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2],
        "ttft_s": 3,
        "e2e_s": 12,
        "compliance": 0.95,
        "request_timeout_s": 30,
        "dispatch_lag_limit_s": 0.05,
    },
}


def fixture(rate=1, phase="measured", *, ttft=1, e2e=2, lag=0):
    count = 16 if phase == "warmup" else 128
    hashes = [hashlib.sha256(f"row {index}".encode()).hexdigest() for index in range(count)]
    rows = []
    for index, digest in enumerate(hashes):
        arrival = index / rate
        sent = arrival + lag
        first = arrival + ttft
        terminal = arrival + e2e
        length = 512 if index % 2 == 0 else 2048
        rows.append({
            "index": index, "input_tokens": length, "prompt_sha256": digest,
            "scheduled_s": arrival, "sent_s": sent, "send_lag_s": sent - arrival,
            "first_token_s": first, "last_token_s": first, "stream_done_s": terminal,
            "response_end_s": terminal, "end_s": terminal, "status": "ok",
            "output_tokens": 128, "server_input_tokens": length,
            "http_status": 200, "finish_reason": "length",
        })
    return dict(schema_version=1, model=MODEL, phase=phase, rate_rps=rate, rows=rows), hashes


def summarize(receipt, hashes):
    return summarize_phase(receipt, CONFIG, MODEL, receipt["rate_rps"], receipt["phase"], hashes)


def no_stream(row, status, *, unsent=False):
    row.update(status=status, first_token_s=None, last_token_s=None, stream_done_s=None,
               response_end_s=None, output_tokens=0)
    for key in ("finish_reason", "server_input_tokens", "http_status"):
        del row[key]
    if unsent:
        row.update(sent_s=None, send_lag_s=None)
    if status == "rejected":
        row["http_status"] = 429


class ServingCapacityPhaseTests(unittest.TestCase):
    def test_inclusive_limits_failures_in_denominator_and_drained_throughput(self):
        raw, hashes = fixture(0.75, ttft=3, e2e=12)
        for i in (0, 2, 4, 6):
            row = raw["rows"][i]
            no_stream(row, "timeout", unsent=True)
            row["end_s"] = row["scheduled_s"] + 30
        no_stream(raw["rows"][1], "rejected")
        raw["rows"][3].update(status="error", output_tokens=5, finish_reason="stop",
                              stream_done_s=None, response_end_s=None)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            summary = summarize(json.loads(path.read_text(encoding="utf-8")), hashes)
        duration = 127 / 0.75 + 12
        self.assertEqual(summary["offered"], 128)
        self.assertEqual(summary["sent"], 124)
        self.assertEqual(summary["completed"], 122)
        self.assertEqual(summary["compliant"], 122)
        self.assertEqual(summary["status_counts"], {"ok": 122, "error": 1, "timeout": 4, "rejected": 1})
        self.assertEqual(summary["slo_status"], "pass")
        self.assertEqual(summary["compliance_fraction"], 122 / 128)
        self.assertEqual(summary["offered_rps"], 0.75)
        self.assertEqual(summary["duration_s"], duration)
        self.assertEqual(summary["completed_rps"], 122 / duration)
        self.assertEqual(summary["compliant_rps"], 122 / duration)
        self.assertAlmostEqual(summary["ttft"]["p95_s"], 3)
        self.assertAlmostEqual(summary["e2e"]["p95_s"], 12)
        self.assertTrue(summary["generator_limited"])
        self.assertIn("missing_sends", summary["limitations"])
        short, long = (summary["by_input_tokens"][length] for length in ("512", "2048"))
        self.assertEqual((short["compliant"], long["compliant"]), (60, 62))
        self.assertEqual(short["offered_rps"], 0.375)
        self.assertEqual(long["completed_rps"], 62 / duration)

        row = raw["rows"][7]
        row["last_token_s"] = row["scheduled_s"] + 12
        row["end_s"] = row["response_end_s"] = math.nextafter(row["end_s"], math.inf)
        summary = summarize(raw, hashes)
        self.assertEqual((summary["completed"], summary["compliant"]), (122, 121))
        self.assertEqual(summary["slo_status"], "fail")

    def test_terminal_error_can_extend_duration_and_empty_success_percentiles(self):
        raw, hashes = fixture(2, "warmup")
        for row in raw["rows"]:
            no_stream(row, "error", unsent=True)
        raw["rows"][0]["end_s"] = 40
        summary = summarize(raw, hashes)
        self.assertEqual(summary["offered"], 16)
        self.assertEqual(summary["offer_duration_s"], 8)
        self.assertEqual(summary["duration_s"], 40)
        self.assertEqual(summary["completed_rps"], 0)
        for metric in ("ttft", "e2e", "dispatch_lag"):
            self.assertEqual(summary[metric], {"p50_s": None, "p95_s": None})

    def test_dispatch_lag_uses_strict_limit_and_slo_uses_arrival(self):
        raw, hashes = fixture(2, ttft=3, e2e=4, lag=0.05)
        summary = summarize(raw, hashes)
        self.assertFalse(summary["generator_limited"])
        self.assertAlmostEqual(summary["dispatch_lag"]["p95_s"], 0.05)
        for row in raw["rows"]:
            row["sent_s"] = row["scheduled_s"] + 0.06
            row["send_lag_s"] = row["sent_s"] - row["scheduled_s"]
            row["first_token_s"] = row["last_token_s"] = row["scheduled_s"] + 3.01
        summary = summarize(raw, hashes)
        self.assertTrue(summary["generator_limited"])
        self.assertEqual(summary["slo_status"], "fail")

    def test_queue_uses_left_limit_at_exact_quarter_boundaries(self):
        rate = 1.25
        raw, hashes = fixture(rate, ttft=0, e2e=0)
        for cutoff, count in ((64, 1), (96, 2), (128, 3)):
            for row in raw["rows"][cutoff - count:cutoff]:
                row["end_s"] = row["response_end_s"] = cutoff / rate
        # At Q3 this arrival is excluded, even though its terminal is later.
        raw["rows"][96]["end_s"] += 1
        raw["rows"][96]["response_end_s"] = raw["rows"][96]["end_s"]
        # This scheduled request remains pending just before Q4, but was unsent.
        no_stream(raw["rows"][126], "timeout", unsent=True)
        # Sending exactly at Q4 still excludes the request from Q4 in-flight.
        last = raw["rows"][127]
        last.update(sent_s=128 / rate, first_token_s=128 / rate, last_token_s=128 / rate,
                    stream_done_s=128 / rate, send_lag_s=128 / rate - last["scheduled_s"])
        summary = summarize(raw, hashes)
        quarters = summary["queue_quarters"]
        self.assertEqual([q["boundary_s"] for q in quarters], [32/rate, 64/rate, 96/rate, 128/rate])
        self.assertEqual([q["pending"] for q in quarters], [0, 1, 2, 3])
        self.assertEqual([q["in_flight"] for q in quarters], [0, 1, 2, 1])
        self.assertTrue(summary["queue_growth"])
        self.assertEqual(summary["slo_status"], "pass")
        self.assertEqual(summary["duration_s"], 128 / rate)

    def test_rejects_partial_phase_and_malformed_rows(self):
        raw, hashes = fixture()
        raw["rows"].pop()
        with self.assertRaisesRegex(ValueError, "128 terminal rows"):
            summarize(raw, hashes)

        for key, value in (
            ("index", True), ("input_tokens", 2048), ("prompt_sha256", "0" * 64),
            ("scheduled_s", 0.01), ("status", "success"), ("end_s", math.inf),
            ("first_token_s", math.nan), ("first_token_s", None), ("sent_s", None),
            ("sent_s", 1.5), ("sent_s", -0.1), ("send_lag_s", -0.1), ("send_lag_s", 0.1),
            ("end_s", 0.9), ("output_tokens", 127),
            ("output_tokens", True), ("last_token_s", 0.9), ("last_token_s", 2.1),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                raw, hashes = fixture()
                raw["rows"][0][key] = value
                summarize(raw, hashes)
        for key in ("sent_s", "first_token_s", "end_s"):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, key):
                raw, hashes = fixture()
                del raw["rows"][0][key]
                summarize(raw, hashes)
        raw, hashes = fixture()
        with self.assertRaisesRegex(ValueError, "expected SHA-256"):
            summarize(raw, hashes[:-1])
        for key, value in (("model", "alien"), ("schema_version", True), ("rate_rps", 0.75),
                           ("phase", "warmup")):
            with self.subTest(identity=key), self.assertRaisesRegex(ValueError, key):
                summarize_phase(dict(raw, **{key: value}), CONFIG, MODEL, 1, "measured", hashes)

    def test_success_requires_all_stream_proof_fields(self):
        for key in ("last_token_s", "stream_done_s", "response_end_s", "finish_reason",
                    "server_input_tokens", "http_status"):
            for missing in (True, False):
                with self.subTest(key=key, missing=missing), self.assertRaisesRegex(ValueError, key):
                    raw, hashes = fixture()
                    raw["rows"][0][key] = None
                    if missing:
                        del raw["rows"][0][key]
                    summarize(raw, hashes)

    def test_rejects_success_proof_contradictions(self):
        for key, value in (
            ("http_status", 429), ("http_status", 500), ("http_status", 200.0),
            ("finish_reason", "stop"), ("server_input_tokens", 2048),
            ("stream_done_s", 0.9), ("stream_done_s", 2.1),
            ("response_end_s", 1.9), ("response_end_s", math.inf), ("end_s", 2.1),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                raw, hashes = fixture()
                raw["rows"][0][key] = value
                summarize(raw, hashes)
        raw, hashes = fixture(e2e=30)
        with self.assertRaisesRegex(ValueError, "request timeout"):
            summarize(raw, hashes)

    def test_failed_stream_states_remain_in_denominator(self):
        raw, hashes = fixture()
        # A cut after [DONE] can preserve tokens and DONE while EOF is absent.
        raw["rows"][0].update(status="error", response_end_s=None)
        # A timeout during cleanup can preserve every marker, with a later terminal.
        raw["rows"][1].update(status="timeout", end_s=31)
        # The client can record HTTP status before detecting that the send mark is missing.
        no_stream(raw["rows"][2], "error", unsent=True)
        raw["rows"][2]["http_status"] = 200
        # Early errors can preserve mismatched server usage and partial token counts.
        raw["rows"][3].update(status="error", server_input_tokens=512, output_tokens=9,
                              stream_done_s=None, response_end_s=None)
        summary = summarize(raw, hashes)
        self.assertEqual(summary["offered"], 128)
        self.assertEqual(summary["completed"], 124)
        self.assertEqual(summary["compliant"], 124)
        self.assertTrue(summary["generator_limited"])
        raw["rows"][2]["sent_s"] = -0.1
        with self.assertRaisesRegex(ValueError, "sent_s"):
            summarize(raw, hashes)


def cell(rate, block, result="pass", *, lag=0):
    raw, hashes = fixture(rate, ttft=2 if result == "pass" else 4, e2e=5, lag=lag)
    return {"block": block, "model_key": MODEL, "group": "four", "rate_rps": rate,
            "summary": summarize(raw, hashes)}


class ServingCapacityFrontierTests(unittest.TestCase):
    def test_bracket_and_censoring_from_three_paired_blocks(self):
        rates = CONFIG["capacity"]["rates_rps"]
        for outcome in ("bracketed", "left_censored", "right_censored"):
            with self.subTest(outcome=outcome):
                cells = []
                for rate in rates:
                    for block in (1, 2, 3):
                        passed = outcome == "right_censored" or (
                            outcome == "bracketed" and (rate <= 1 or block == 2)
                        )
                        cells.append(cell(rate, block, "pass" if passed else "fail", lag=0.06))
                frontiers = capacity_frontiers(list(reversed(cells)), CONFIG)
                self.assertEqual(len(frontiers), 6)
                frontier = frontiers[0]
                self.assertEqual(frontier["status"], outcome)
                expected = {
                    "bracketed": (1, 1.25), "left_censored": (None, 0.25),
                    "right_censored": (2, None),
                }
                self.assertEqual((frontier["highest_passing_rate_rps"],
                                  frontier["first_higher_failing_rate_rps"]), expected[outcome])
                self.assertIn("dispatch_lag_p95_exceeded", frontier["limitations"])
                self.assertEqual(frontiers[-1]["status"], "indeterminate")

    def test_partial_rates_and_nonmonotonic_blocks_remain_visible(self):
        cells = [cell(0.25, block) for block in (1, 2, 3)]
        cells += [cell(0.5, 2), cell(0.75, 1, "fail")]
        cells += [cell(1, block) for block in (1, 2, 3)]
        cells += [cell(1.5, 3, "fail"), cell(2, 1)]
        frontier = capacity_frontiers(cells, CONFIG)[0]
        self.assertEqual(frontier["status"], "indeterminate")
        self.assertEqual(frontier["highest_passing_rate_rps"], 1)
        self.assertEqual(frontier["first_higher_failing_rate_rps"], 1.5)
        self.assertEqual(frontier["nonmonotonic_blocks"], [1])
        self.assertEqual(frontier["incomplete_rates_rps"], [0.5, 0.75, 1.25, 1.5, 2])
        self.assertEqual([r["slo_status"] for r in frontier["rates"]],
                         ["pass", "indeterminate", "fail", "pass", "indeterminate", "fail",
                          "indeterminate"])
        self.assertEqual(frontier["rates"][2]["missing_blocks"], [2, 3])

    def test_within_block_reversal_can_exist_with_monotonic_mix_outcomes(self):
        cells = [
            cell(rate, block, "fail" if block == 1 or rate == 0.25 else "pass")
            for rate in CONFIG["capacity"]["rates_rps"] for block in (1, 2, 3)
        ]
        frontier = capacity_frontiers(cells, CONFIG)[0]
        self.assertEqual(frontier["status"], "indeterminate")
        self.assertEqual(frontier["nonmonotonic_blocks"], [2, 3])

    def test_model_and_topology_cannot_supply_each_others_missing_blocks(self):
        other_model = "llama31-8b"
        first = cell(0.25, 1)
        pair = dict(cell(0.25, 2), group="pair")
        llama = dict(cell(0.25, 3), model_key=other_model)
        llama["summary"] = dict(llama["summary"], model_key=other_model)
        results = {(row["model_key"], row["group"]): row
                   for row in capacity_frontiers([first, pair, llama], CONFIG)}
        for model, group, block in ((MODEL, "four", 1), (MODEL, "pair", 2), (other_model, "four", 3)):
            frontier = results[model, group]
            self.assertIsNone(frontier["highest_passing_rate_rps"])
            self.assertEqual(frontier["rates"][0]["passing_blocks"], [block])
            self.assertEqual(frontier["rates"][0]["slo_status"], "indeterminate")

    def test_rejects_duplicate_and_mismatched_cells(self):
        observed = cell(0.25, 1)
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            capacity_frontiers([observed, observed], CONFIG)
        with self.assertRaisesRegex(ValueError, "campaign grid"):
            capacity_frontiers([dict(observed, block=4)], CONFIG)
        with self.assertRaisesRegex(ValueError, "matching measured"):
            capacity_frontiers([dict(observed, rate_rps=0.5)], CONFIG)
        warmup, hashes = fixture(0.25, "warmup")
        with self.assertRaisesRegex(ValueError, "matching measured"):
            capacity_frontiers([dict(observed, summary=summarize(warmup, hashes))], CONFIG)
