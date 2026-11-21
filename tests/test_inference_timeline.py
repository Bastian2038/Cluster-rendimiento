import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infraestructura import inference_diagnosis_report as diagnosis
from infraestructura import inference_timeline as timeline
from tests.test_inference_metrics import fixture


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = fixture(duration=8, concurrency=8)
        self.raw.update(start_times=[500 + (i // 8) * 2 for i in range(32)], max_concurrency=8)
        self.samples = [{"id": f"512-{i:03d}"} for i in range(16, 48)]

    def test_reconstructs_concurrent_requests_and_slot_time(self):
        row = timeline.client_timeline(self.raw, self.samples, 512, 8)
        self.assertEqual(row["peak_in_flight"], 8)
        self.assertAlmostEqual(row["t32_minus_span_s"], 0.63)
        self.assertAlmostEqual(row["span_s"], 7.37)
        self.assertEqual(row["first_start_beyond_initial_slots_s"], 2)
        self.assertEqual(row["last_request_start_s"], 6)
        self.assertAlmostEqual(row["longest_pause"]["duration_s"], 0.01)
        self.assertEqual(row["requests"][8]["prompt_id"], "512-024")
        self.assertAlmostEqual(row["requests"][8]["first_token_s"], 2.1)
        self.assertAlmostEqual(row["ttft_slot_seconds_div_c"] + row["stream_slot_seconds_div_c"] +
                               row["remaining_slot_seconds_div_c"], 8)

    def test_refuses_incomplete_intervals_overbooking_and_duplicate_ids(self):
        for raw, ids in (
            (dict(self.raw, itls=[[0.1] * 126] * 32), self.samples),
            (dict(self.raw, start_times=[0] * 32), self.samples),
            (dict(self.raw, start_times=[float("nan")] * 32), self.samples),
            (self.raw, [self.samples[0]] * 32),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                timeline.client_timeline(raw, ids, 512, 8)

    def test_only_approved_pairs_and_intact_intervals_contribute(self):
        tl = timeline.client_timeline(self.raw, self.samples, 512, 8)
        cell = {"model_key": "model", "block": 1, "input_tokens": 512, "concurrency": 8,
                "timeline": dict(tl, valid=True)}
        a, b = dict(cell, group="four"), dict(cell, group="pair")
        pair = {"model": "model", "block": 1, "input_tokens": 512, "concurrency": 8, "s4_2": 1.0}
        self.assertEqual(timeline.paired_timelines([a, b], []), [])
        self.assertEqual(timeline.paired_timelines([a, b], [pair])[0]["four_minus_two_s"]["t32_s"], 0)
        self.assertEqual(timeline.paired_timelines([a, dict(b, timeline={"valid": False})], [pair]), [])
        with self.assertRaises(ValueError):
            timeline.paired_timelines([a, b], [dict(pair, s4_2=1.1)])

    def test_report_set_never_overwrites_an_existing_derivative(self):
        with tempfile.TemporaryDirectory() as directory:
            md = Path(directory) / "resumen.md"
            md.write_text("existente")
            with self.assertRaises(FileExistsError):
                diagnosis.write_outputs(md, {"campaign_id": "fixture", "capture_commit": "abc",
                                            "status": "partial", "primary_cells": [],
                                            "timeline_contrasts_4_2": []})
            self.assertEqual(md.read_text(), "existente")
            self.assertFalse(md.with_suffix(".json").exists())

    def test_phase_contrasts_only_for_verified_4_2_blocks_and_all_phases(self):
        identity = dict(model_key="m", block=1, input_tokens=2048, concurrency=8)
        pair = dict(model="m", block=1, input_tokens=2048, concurrency=8, s4_2=0.9)
        def cell(group, mean, valid=True):
            stats = {phase: {"mean_s": mean} for phase in diagnosis.PHASES}
            return dict(identity, group=group, server_phases=dict(valid=valid, phases=stats))
        four = cell("four", 0.375)
        self.assertEqual(diagnosis.phase_contrasts([four, cell("pair", 0.25)], [pair])[0]
                         ["four_minus_two_mean_s"]["queue"], 0.125)
        self.assertEqual(diagnosis.phase_contrasts([four, cell("pair", 0.25, False)], [pair]), [])
        only_two_one = {k: v for k, v in pair.items() if k != "s4_2"}
        only_two_one["s2_1"] = 1.2
        self.assertEqual(diagnosis.phase_contrasts([four], [only_two_one]), [])

    def test_off_on_requires_same_run_and_verified_observations(self):
        off = dict(run_id="r", model_key="m", group="four", block=4, input_tokens=2048,
                   concurrency=8, mode="off", t32_s=8, client_admission={"valid": True})
        on = dict(off, mode="on", t32_s=10, profile_control={"valid": True},
                  profile_coverage={"valid": True}, trace={"valid": True})
        self.assertEqual(diagnosis.calibrate([off, on])[0]["delta_percent"], 25)
        self.assertEqual(diagnosis.calibrate([off, dict(on, trace={"valid": False})])[0]["delta_percent"], 25)
        for rec in (dict(on, run_id="s"), dict(on, profile_control={"valid": False}),
                    dict(on, profile_coverage={"valid": False}),
                    dict(on, client_admission={"valid": False})):
            for pair in diagnosis.calibrate([off, rec]):
                self.assertFalse(pair["valid"])
                self.assertNotIn("delta_percent", pair)
        with self.assertRaises(ValueError):
            diagnosis.calibrate([off, off])
        pairs = diagnosis.calibrate([off, on, dict(off, run_id="p", group="pair", t32_s=9),
                                     dict(on, run_id="p", group="pair", t32_s=9)])
        comparison = diagnosis.compare_profile_order(pairs, [dict(model_key="m", input_tokens=2048,
            concurrency=8, four_minus_two_s={"t32_s": -1})])[0]
        self.assertTrue(comparison["order_reversed_when_on_requested"])
        self.assertEqual(comparison["primary_delta_s"], -1)
        self.assertEqual(diagnosis.compare_profile_order(pairs[:1], []), [])
        untraced = diagnosis.compare_profile_order([dict(p, rank_traces_verified=False) for p in pairs], [])
        self.assertTrue(untraced[0]["order_reversed_when_on_requested"])
        self.assertEqual(untraced[0]["rank_sets_verified"], {"four": False, "pair": False})
        emitted = "\n".join(diagnosis.instrument_tables(dict(primary_cells=[], auxiliary_cells=[],
            off_on_pairs=[], profile_order_4_2=untraced, phase_contrasts_4_2=[], primary_telemetry=[])))
        self.assertIn("cambio de signo) | no / no", emitted)
        self.assertNotIn("la captura altera", emitted)

    def test_telemetry_follows_the_participant_and_measured_window_in_a_partial_run(self):
        rank = dict(hostname="spark", node_id="pegasus", rank=0, gpu_uuid="fixture")
        with tempfile.TemporaryDirectory() as directory, patch.object(
                diagnosis.manifest, "load_run", return_value={"status": "failed", "participants": [rank]}):
            path = Path(directory) / "raw/nodes/pegasus/logs/telemetry.jsonl"
            path.parent.mkdir(parents=True)
            stamps = [f"2026-11-21T12:00:0{i}Z" for i in range(4)]
            path.write_text("\n".join(json.dumps(dict(hostname="spark", timestamp_utc=t,
                monotonic_ns=i * 10**9, gpus=[{"uuid": "fixture"}], memory={"MemAvailable_kib": 12 * 1024**2}))
                for i, t in enumerate(stamps)) + '\n{"cola":')
            cell = dict(model_key="m", group="single", block=1, cell="2048-c8", run_dir=directory,
                        measurement_started_at=stamps[1], measurement_ended_at=stamps[2])
            row, = diagnosis.primary_telemetry([cell])
            self.assertEqual((row["sample_count"], row["node_id"], row["rank"]), (2, "pegasus", 0))
            self.assertEqual(row["source_sha256"], diagnosis.manifest.sha256_file(path))
