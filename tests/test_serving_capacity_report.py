"""CPU-only fixtures for the serving campaign reader."""

import asyncio
from contextlib import ExitStack, asynccontextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from infraestructura import (
    inference, inference_evidence, manifest, serving_capacity,
    serving_capacity_client as client, serving_capacity_metrics as metrics,
    serving_capacity_protocol as protocol, serving_capacity_report as report,
    serving_capacity_runner as runner,
)
from tests.test_inference import fixture_campaign_inputs, ready_probe, write_fixture_roster
from tests.test_serving_capacity_client import Clock, Response, stream as sse_stream


COMMIT = "a" * 40
START = datetime(2026, 11, 22, tzinfo=timezone.utc)
FIRST_POINT = ((3, 67), (73, 585))
CLOCK_OFFSET = 10_000


def stamp(seconds):
    return (START + timedelta(seconds=seconds)).isoformat()


class ServingCapacityReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config, self.inventory, _ = protocol.load_config()
        self.jobs = list(protocol.jobs(self.config))
        self.preflight = {"status": "ready_for_smoke", "errors": [], "nodes": [
            ready_probe(self.config, self.inventory, node) for node in self.inventory["nodes"]]}
        self.git = {"commit": COMMIT, "dirty": False}
        self.directory = self.root / "raw/serving-capacity-campaign/id"
        self.plan = {"protocol": self.config,
                     "jobs": protocol.planned_jobs(self.config, self.inventory)}

    def seal(self, runs=(), complete=False, *, extra_attempt=None, limit=3600, session_end=None,
             intent_only=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        state = "complete" if complete else "partial"
        source = self.root / "reference-preflight.json"
        archived = self.directory / "reference/preflight.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, archived)
        attempts = []
        run_end = -2
        observed = -2
        confirmed = True
        for ref in runs:
            started_at = manifest.load_run(ref["run_dir"])["started_at"]
            started = (datetime.fromisoformat(started_at) - START).total_seconds()
            run_end = (datetime.fromisoformat(manifest.load_run(ref["run_dir"])["ended_at"]) - START).total_seconds()
            attempt = {"job": ref["job"], "run_id": Path(ref["run_dir"]).name, "run_dir": ref["run_dir"],
                       "status": "sealed", "occupation": ref["occupation"], "errors": ref["errors"],
                       "manifest_sha256": ref["manifest_sha256"]}
            attempts.append(self.write_attempt(attempt, started - 0.5))
            if ref["occupation"] is None:
                confirmed = False
            else:
                observed = ref["occupation"]["observed_until_monotonic_s"] - CLOCK_OFFSET
        if extra_attempt:
            attempts.append(self.write_attempt(extra_attempt, run_end + 0.5, write_state=not intent_only))
        end = run_end + 1 if session_end is None else session_end
        confirmed = confirmed and (extra_attempt is None or extra_attempt["status"] == "not_started")
        if not confirmed or not attempts:
            observed = end
        index = {
            "campaign_id": "id", "protocol": self.config, "git": self.git, "runs": list(runs),
            "status": state, "pending_jobs": [job["name"] for job in self.jobs[len(attempts):]],
            "errors": [] if complete else ["presupuesto agotado"],
            "input_hashes": self.hashes, "attempts": attempts,
            "hardware": protocol.verify_hardware(self.config, archived),
            "started_at": stamp(-2), "ended_at": stamp(end),
            "started_monotonic_s": CLOCK_OFFSET - 2, "ended_monotonic_s": CLOCK_OFFSET + end,
            "duration_seconds": end + 2, "effective_session_limit_seconds": limit,
            "session_limit_exceeded": end + 2 > limit,
            "occupation": {
                "started_at": stamp(-2), "started_monotonic_s": CLOCK_OFFSET - 2,
                "ended_at": stamp(observed) if confirmed else None,
                "ended_monotonic_s": CLOCK_OFFSET + observed if confirmed else None,
                "observed_until_at": stamp(observed), "observed_until_monotonic_s": CLOCK_OFFSET + observed,
                "duration_seconds": observed + 2, "removal_confirmed": confirmed,
                "limit_exceeded": observed + 2 > limit,
            },
        }
        for name, payload in (("campaign", index), ("plan", self.plan), ("preflight", self.preflight)):
            inference.write_new(self.directory / f"{name}.json", payload)
        self.mark(state.upper())

    def write_attempt(self, attempt, requested, *, write_state=True):
        job = attempt["job"]
        intent_path = f"attempts/{job}.json"
        intent = {key: attempt[key] for key in ("job", "run_id", "run_dir")}
        intent.update(schema_version=1, campaign_id="id", requested_at=stamp(requested),
                      requested_monotonic_s=CLOCK_OFFSET + requested)
        inference.write_new(self.directory / intent_path, intent)
        attempt.update(intent_path=intent_path, intent_sha256=manifest.sha256_file(self.directory / intent_path))
        if write_state:
            state_path = f"attempts/{job}-state.json"
            inference.write_new(self.directory / state_path, attempt)
            attempt.update(state_path=state_path, state_sha256=manifest.sha256_file(self.directory / state_path))
        return attempt

    def mark(self, state):
        marker = {f"{name}_sha256": manifest.sha256_file(self.directory / f"{name}.json")
                  for name in ("campaign", "plan", "preflight")}
        (self.directory / state).write_text(json.dumps(marker), encoding="utf-8")

    def inputs(self):
        packages = json.loads(self.preflight["nodes"][0]["container_probe"]["stdout"])["packages"]
        inputs, _ = fixture_campaign_inputs(self.root, self.config, packages)
        for name in inference_evidence.runner_sources(protocol.PROTOCOL):
            inputs.setdefault(name, self.root / "prepared" / name)
            if not inputs[name].exists():
                inputs[name].write_text(f"# Fixture {name}\n", encoding="utf-8")

        historical_hashes = {name: manifest.sha256_file(path) for name, path in inputs.items()}
        historical = {**self.config, "measurement": {"warmup_requests": 16, "measured_requests": 32}}
        inputs["reference/campaign.json"] = self.root / "reference.json"
        inference.write_new(inputs["reference/campaign.json"], {
            "campaign_id": protocol.HISTORICAL_ID, "status": "complete", "errors": [],
            "pending_jobs": [], "protocol": historical, "input_hashes": historical_hashes,
        })
        digest = manifest.sha256_file(inputs["reference/campaign.json"])
        patch = mock.patch.object(protocol, "HISTORICAL_SHA256", digest)
        patch.start()
        self.addCleanup(patch.stop)
        self.config["reference"]["historical_campaign_sha256"] = digest
        inputs["reference/preflight.json"] = self.root / "reference-preflight.json"
        inference.write_new(inputs["reference/preflight.json"], self.preflight)
        digest = manifest.sha256_file(inputs["reference/preflight.json"])
        patch = mock.patch.object(protocol, "HISTORICAL_PREFLIGHT_SHA256", digest)
        patch.start()
        self.addCleanup(patch.stop)
        self.config["reference"]["historical_preflight_sha256"] = digest
        self.config["reference"]["historical_jsonl_sha256"] = {
            name: digest for name, digest in historical_hashes.items() if name.endswith(".jsonl")}
        for name, payload in (("cluster", self.inventory), ("protocol", self.config)):
            inputs[name] = self.root / f"{name}.json"
            inference.write_new(inputs[name], payload)
        self.hashes = {name: manifest.sha256_file(path) for name, path in inputs.items()}
        return inputs

    def start_fixture_run(self, job, inputs, run_id, started=0):
        model = job["model"]
        with (mock.patch.object(manifest, "capture_git", return_value=self.git),
              mock.patch.object(manifest, "capture_system", return_value={"hostname": "fixture"}),
              mock.patch.object(manifest, "capture_nvidia", return_value={}),
              mock.patch.object(manifest, "capture_packages", return_value={}),
              mock.patch.object(manifest, "utc_now", return_value=stamp(started))):
            return manifest.start_run(
                kind="inference-serving", run_id=run_id, model_id=model["id"],
                model_revision=model["revision"], tokenizer_id=model["tokenizer_id"],
                tokenizer_revision=model["tokenizer_revision"], inputs=inputs,
                requested_config={"protocol": self.config, "job": job, "campaign_id": "id"},
                results_root=self.root / "raw",
            )

    def write_occupation(self, path, start, end, *, limit_exceeded=False, removal_confirmed=True):
        receipt = {"started_at": stamp(start), "started_monotonic_s": CLOCK_OFFSET + start,
                   "ended_at": stamp(end) if removal_confirmed else None,
                   "ended_monotonic_s": CLOCK_OFFSET + end if removal_confirmed else None,
                   "observed_until_at": stamp(end), "observed_until_monotonic_s": CLOCK_OFFSET + end,
                   "duration_seconds": end - start, "removal_confirmed": removal_confirmed,
                   "limit_exceeded": limit_exceeded}
        inference.write_new(path / "raw/occupation.json", receipt)
        return receipt

    def run_fixture(self, *, complete=False, corrupt_measured=False, windows=(FIRST_POINT,),
                    omit_tokenization=False, omit_quality_at=None, omit_reference_preflight=False,
                    omit_occupation=False):
        inputs = self.inputs()
        job = self.jobs[0]
        model = job["model"]
        if omit_reference_preflight:
            inputs.pop("reference/preflight.json")
        path = self.start_fixture_run(job, inputs, "fixture")
        for name, source in inputs.items():
            target = path / "raw/inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        inference.write_new(path / "raw/preflight.json", self.preflight)
        if not omit_tokenization:
            inference.write_new(path / "raw/tokenization-verified.json", {"verified": True, "profiles": [512, 2048]})
        nodes = self.inventory["groups"][job["group"]]
        write_fixture_roster(path, model, nodes, self.preflight)
        ranks = [inference.read_json(path / f"raw/participants/rank-{rank}.json") for rank in range(len(nodes))]
        inference.write_new(path / "raw/startup.json", {"ranks": ranks})
        inference.write_new(path / "raw/cleanup.json", [
            {"name": f"ipre-id-{job['name']}-{node}", "status": "confirmed_absent"}
            for node in nodes])
        end = windows[-1][1][1]
        logs = self.write_node_evidence(path, ranks, end)
        cells = [self.write_point(path, job, coordinate["rate_rps"], spans, logs, corrupt_measured)
                 for coordinate, spans in zip(job["cells"], windows)]
        for cell in cells:
            if cell["rate_rps"] != omit_quality_at:
                inference.write_new(path / f"raw/diagnostic/{cell['cell']}/quality.json",
                                    {"valid": True, "issues": []})
        errors = [] if complete else ["presupuesto agotado"]
        inference.write_new(path / "metrics/assessment.json",
                            {"cells": cells, "valid": complete, "errors": errors})
        status = "complete" if complete else "failed"
        occupation = None if omit_occupation else self.write_occupation(path, 1, end + 5)
        with mock.patch.object(manifest, "utc_now", return_value=stamp(end + 25)):
            manifest.finalize_run(path, status=status, errors=errors)
        return {"job": job["name"], "group": job["group"], "block": job["block"],
                "model_key": model["key"], "run_dir": str(path), "status": status,
                "manifest_sha256": manifest.sha256_file(path / "manifest.json"), "errors": errors,
                "occupation": occupation}

    def write_point(self, path, job, rate, windows, logs, corrupt_measured):
        name = protocol.point_name(rate)
        model = job["model"]["key"]
        evidence = path / "raw/serving-capacity" / name
        for phase, (start, end) in zip(protocol.PHASES, windows):
            prompts = client.expected_prompts(
                path / "raw/inputs/workload" / model, phase,
                self.config["measurement"][f"{phase}_requests"])
            lengths = {prompt["prompt"]: prompt["input_tokens"] for prompt in prompts}
            clock = Clock()
            clock.now = 0.0

            @asynccontextmanager
            async def transport(payload, sent):
                sent()
                yield Response(sse_stream(lengths[payload["prompt"]]))

            receipt = asyncio.run(client.run_phase(
                model, phase, rate, prompts, self.config["capacity"]["request_timeout_s"],
                transport, clock=clock, sleep=clock.sleep))
            summary = metrics.summarize_phase(receipt, self.config, model, rate, phase,
                                              [prompt["prompt_sha256"] for prompt in prompts])
            self.assertEqual(summary["completed"], len(prompts))
            if corrupt_measured and phase == "measured":
                receipt["rows"][0]["finish_reason"] = "stop"
            inference.write_new(path / f"raw/nodes/pegasus/metrics/{name}/{phase}.json", receipt)
            window = {"cell": name, "phase": phase, "measurement_started_at": stamp(start),
                      "measurement_ended_at": stamp(end), "started_monotonic_s": start,
                      "ended_monotonic_s": end, "client_returncode": 0, "client_timed_out": False}
            inference.write_new(evidence / f"{phase}-window.json", window)
            inference.write_new(evidence / f"{phase}-summary.json", summary)
            port = self.config["runtime"]["api_port"]
            for side, instant in (("before", start - 1), ("after", end + 1)):
                inference.write_new(evidence / f"{phase}-{side}.json", {
                    "valid": True, "drained": True, "http_status": 200, "health_http_status": 200,
                    "model_key": model,
                    "models": {"data": [{"id": model}]}, "port": port,
                    "started_monotonic_s": instant, "finished_monotonic_s": instant,
                    "body": "".join(
                        f'# TYPE vllm:num_requests_{gauge} gauge\n'
                        f'vllm:num_requests_{gauge}{{model_name="{model}",engine="0"}} 0\n'
                        for gauge in ("running", "waiting")),
                    "process_identity": {"source": "listener-procfs", "boot_id": "pegasus",
                                         "pid": 42, "start_ticks": 88, "socket_inode": 93, "port": port,
                                         "checked_before_monotonic_s": instant,
                                         "checked_after_monotonic_s": instant},
                })
        cell = {**summary, "rate_rps": rate, "cell": name, "status": "valid",
                "measurement_started_at": window["measurement_started_at"],
                "measurement_ended_at": window["measurement_ended_at"]}
        inference.write_new(path / f"raw/diagnostic/{name}/execution-window.json", {
            "cell": name, "measurement_started_at": cell["measurement_started_at"],
            "measurement_ended_at": cell["measurement_ended_at"], "logs": logs})
        return cell

    def write_node_evidence(self, path, ranks, end):
        logs = {}
        for rank in ranks:
            node, index = rank["node_id"], rank["rank"]
            dest = path / "raw/nodes" / node / "logs"
            dest.mkdir(parents=True, exist_ok=True)
            names = [f"worker-rank{index:02d}.log"] + (["server.log"] if node == "pegasus" else [])
            logs[node] = {filename: [0, 0] for filename in names}
            for filename in names:
                (dest / filename).touch()
            with (dest / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
                for second in range(0, end + 7, 2):
                    sample = {"hostname": rank["hostname"], "timestamp_utc": stamp(second),
                              "monotonic_ns": second * 10**9, "memory": {"MemAvailable_kib": 9 * 2**20},
                              "gpus": [{"uuid": rank["gpu_uuid"]}]}
                    stream.write(json.dumps(sample) + "\n")
        return logs

    def test_partial_keeps_audited_point_and_writes_exclusively(self):
        ref = self.run_fixture()
        self.seal([ref])
        # Original input paths are mutable and play no part in offline verification.
        (self.root / "cluster.json").write_text("alterado", encoding="utf-8")
        data = report.build_report(self.directory)
        self.assertTrue(data["plan_commands_verified"])
        self.assertEqual(len(data["cells"]), 1)
        self.assertEqual(data["cells"][0]["summary"]["compliant"], 128)
        self.assertEqual(len(data["cells"][0]["telemetry"]), 4)
        self.assertEqual(len(data["missing_cells"]), 125)
        self.assertEqual(data["frontiers"][0]["rates"][0]["missing_blocks"], [2, 3])
        output = self.root / "summary.md"
        output.write_text("existente", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            report.write_report(output, data)
        self.assertEqual(output.read_text(encoding="utf-8"), "existente")
        self.assertFalse(output.with_suffix(".json").exists())
        output.unlink()
        self.assertEqual(serving_capacity.main(["report", str(self.directory), "--output", str(output)]), 0)
        self.assertIn("Puntos verificados 1/126", output.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(output.with_suffix(".json").read_text()), data)

    def test_copied_archive_resolves_its_own_run_while_original_exists(self):
        ref = self.run_fixture()
        self.seal([ref])
        original_root = self.directory.parent.parent
        copied_root = self.root / "copy"
        copied_campaign = copied_root / self.directory.relative_to(original_root)
        shutil.copytree(original_root, copied_root)

        original = report.build_report(self.directory)
        self.assertEqual(len(original["cells"]), 1)
        self.assertEqual(report.build_report(copied_campaign), original)
        original_run = Path(ref["run_dir"])
        self.assertTrue(original_run.is_dir())

        copied_manifest = copied_root / original_run.relative_to(original_root) / "manifest.json"
        with copied_manifest.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 del manifiesto incompatible"):
            report.build_report(copied_campaign)

    def test_partial_excludes_sealed_success_with_inconsistent_raw_stream(self):
        self.seal([self.run_fixture(corrupt_measured=True)])
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 0)
        self.assertEqual(len(data["missing_cells"]), 126)
        excluded, = data["excluded_cells"]
        self.assertEqual(excluded["rate_rps"], self.config["capacity"]["rates_rps"][0])
        self.assertIn("finish_reason", excluded["reason"])

    def test_partial_excludes_warmup_before_run_start(self):
        self.seal([self.run_fixture(windows=(((-1, 67), FIRST_POINT[1]),))])
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 0)
        excluded, = data["excluded_cells"]
        self.assertIn("antes del inicio de la corrida", excluded["reason"])

    def test_partial_excludes_warmup_before_previous_point_closes(self):
        # Client windows touch in UTC, but the previous health scrape ends one second later.
        self.seal([self.run_fixture(windows=(FIRST_POINT, ((585, 617), (623, 879))))])
        data = report.build_report(self.directory)
        self.assertEqual([cell["rate_rps"] for cell in data["cells"]], [0.25])
        excluded, = data["excluded_cells"]
        self.assertEqual(excluded["rate_rps"], 0.5)
        self.assertIn("antes del cierre del punto anterior", excluded["reason"])

    def test_partial_requires_archived_historical_preflight(self):
        self.seal([self.run_fixture(omit_reference_preflight=True)])
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 0)
        excluded, = data["excluded_cells"]
        self.assertIn("reference/preflight.json", excluded["reason"])

    def test_partial_excludes_hardware_drift_from_archived_preflight(self):
        ref = self.run_fixture()
        self.preflight["nodes"][0]["gpu"]["stdout"] = "GPU-0, NVIDIA GB10, driver-distinto\n"
        self.seal([ref])
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 0)
        excluded, = data["excluded_cells"]
        self.assertIn("driver difiere", excluded["reason"])

    def test_partial_requires_tokenization_receipt(self):
        self.seal([self.run_fixture(omit_tokenization=True)])
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 0)
        excluded, = data["excluded_cells"]
        self.assertIn("tokenization", excluded["reason"])

    def test_partial_excludes_only_point_with_missing_quality_receipt(self):
        second = ((591, 623), (629, 885))
        self.seal([self.run_fixture(windows=(FIRST_POINT, second), omit_quality_at=0.5)])
        data = report.build_report(self.directory)
        self.assertEqual([cell["rate_rps"] for cell in data["cells"]], [0.25])
        excluded, = data["excluded_cells"]
        self.assertEqual(excluded["rate_rps"], 0.5)
        self.assertIn("quality", excluded["reason"])

    def test_rejects_changed_hash_and_resealed_command(self):
        self.seal([self.run_fixture()])
        self.plan["jobs"][0]["cells"][0]["commands"]["warmup"][0] = "changed"
        (self.directory / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            report.build_report(self.directory)
        self.mark("PARTIAL")
        with self.assertRaisesRegex(ValueError, "comandos"):
            report.build_report(self.directory)

    def test_partial_keeps_verified_point_when_retirement_receipt_is_missing(self):
        self.seal([self.run_fixture(omit_occupation=True)])
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 1)
        self.assertEqual(data["excluded_cells"], [])
        self.assertFalse(data["occupation"]["removal_confirmed"])
        self.assertIsNone(data["occupation"]["ended_monotonic_s"])
        self.assertIsNone(data["post_occupation_seconds"])
        self.assertIn("Falta raw/occupation.json", data["runs"][0]["occupation_diagnostic"])
        output = self.root / "retirement.md"
        report.write_report(output, data)
        self.assertIn("Retirada sin confirmar", output.read_text(encoding="utf-8"))

        campaign_path = self.directory / "campaign.json"
        campaign = inference.read_json(campaign_path)
        campaign["occupation"]["removal_confirmed"] = True
        campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
        self.mark("PARTIAL")
        with self.assertRaisesRegex(ValueError, "ocupación de campaña contradice"):
            report.build_report(self.directory)

    def test_partial_distinguishes_post_retirement_overrun(self):
        ref = self.run_fixture()
        self.seal([ref], limit=600, session_end=615)
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 1)
        self.assertTrue(data["session_limit_exceeded"])
        self.assertFalse(data["occupation"]["limit_exceeded"])
        self.assertEqual(data["post_occupation_seconds"], 25)

    def test_partial_keeps_intent_without_run_or_state(self):
        ref = self.run_fixture()
        job = self.jobs[1]
        path = Path(ref["run_dir"]).with_name("reserved")
        attempt = {"job": job["name"], "run_id": path.name, "run_dir": str(path), "status": "not_started"}
        self.seal([ref], extra_attempt=attempt, intent_only=True)
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 1)
        self.assertEqual(data["attempts"][1]["status"], "not_started")
        self.assertFalse(data["attempts"][1]["state_verified"])
        self.assertEqual(data["pending_jobs"][0], self.jobs[2]["name"])

    def test_partial_excludes_open_attempt_and_checks_state_hash(self):
        ref = self.run_fixture()
        job = self.jobs[1]
        path = self.start_fixture_run(job, {}, "still-open", started=611)
        attempt = {"job": job["name"], "run_id": path.name, "run_dir": str(path), "status": "open",
                   "occupation": None, "errors": ["interrumpido"]}
        self.seal([ref], extra_attempt=attempt, session_end=612)
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 1)
        self.assertFalse(data["occupation"]["removal_confirmed"])
        self.assertEqual(data["attempts"][1]["started_at"], stamp(611))
        state = self.directory / data["attempts"][1]["state_path"]
        with state.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 de state"):
            report.build_report(self.directory)

    def consecutive_seals(self, *, truncated=False):
        windows = []
        rate_list = self.config["capacity"]["rates_rps"]
        start = 3
        for rate in rate_list:
            warm_end = start + math.ceil(self.config["measurement"]["warmup_requests"] / rate)
            measured_start = warm_end + 6
            measured_end = measured_start + math.ceil(self.config["measurement"]["measured_requests"] / rate)
            windows.append(((start, warm_end), (measured_start, measured_end)))
            start = measured_end + 6
        first = self.run_fixture(complete=True, windows=windows)
        job = self.jobs[1]
        started = windows[-1][1][1] + 26
        path = self.start_fixture_run(job, {}, "pending-run", started=started)
        inference.write_new(path / "metrics/assessment.json", {
            "cells": [{"status": "valid", "rate_rps": rate_list[0],
                       "cell": protocol.point_name(rate_list[0])}],
        })
        if truncated:
            (path / "raw/occupation.json").write_text('{"started_at":', encoding="utf-8")
        with mock.patch.object(manifest, "utc_now", return_value=stamp(started + 1)):
            manifest.finalize_run(path, status="failed", errors=["interrumpido"])
        return first, path

    def pending_after_complete_run(self, status, *, truncated=False):
        first, path = self.consecutive_seals(truncated=truncated)
        attempt = {"job": self.jobs[1]["name"], "run_id": path.name, "run_dir": str(path),
                   "status": status, "occupation": None, "errors": ["validación interrumpida"],
                   "manifest_sha256": manifest.sha256_file(path / "manifest.json")}
        end = datetime.fromisoformat(manifest.load_run(path)["ended_at"])
        self.seal([first], extra_attempt=attempt, session_end=(end - START).total_seconds() + 1)
        return path

    def test_interrupted_verification_keeps_previous_points_and_reports_truncated_receipt(self):
        path = self.pending_after_complete_run("verification_pending", truncated=True)
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 7)
        self.assertEqual({cell["group"] for cell in data["cells"]}, {"four"})
        self.assertEqual(len(data["runs"]), 1)
        pending = data["attempts"][1]
        self.assertEqual(pending["status"], "verification_pending")
        self.assertEqual(pending["manifest_status"], "failed")
        self.assertIn("ilegible", pending["occupation_diagnostic"])
        self.assertFalse(data["occupation"]["removal_confirmed"])

        with (path / "raw/occupation.json").open("a", encoding="utf-8") as stream:
            stream.write("0}")
        with self.assertRaisesRegex(ValueError, "SHA-256 o índice del recibo"):
            report.build_report(self.directory)

    def test_legacy_open_attempt_with_terminal_manifest_is_verification_pending(self):
        self.pending_after_complete_run("open")
        data = report.build_report(self.directory)
        self.assertEqual(len(data["cells"]), 7)
        pending = data["attempts"][1]
        self.assertEqual(pending["indexed_status"], "open")
        self.assertEqual(pending["status"], "verification_pending")
        self.assertEqual(pending["manifest_status"], "failed")
        self.assertIn("Falta raw/occupation.json", pending["occupation_diagnostic"])

    def test_run_campaign_report_survives_interrupt_after_second_seal(self):
        first, second = self.consecutive_seals()
        first_path = Path(first["run_dir"])
        archive = first_path / "raw/inputs"
        sources = {path.name: path for path in (first_path, second)}
        output = self.root / "coordinator"
        now = -2.0
        clock = SimpleNamespace(monotonic=lambda: CLOCK_OFFSET + now)
        finish_attempt = runner._finish_attempt
        interruption = "interrupción antes de registrar la segunda referencia"

        def install_sealed_run(*_args, run_id, **_kwargs):
            nonlocal now
            target = output / "inference-serving" / run_id
            shutil.copytree(sources[run_id], target)
            ended = datetime.fromisoformat(manifest.load_run(target)["ended_at"])
            now = (ended - START).total_seconds() + 0.5

        def finish_then_interrupt(attempt, job, config, campaign_id):
            record = finish_attempt(attempt, job, config, campaign_id)
            if job["name"] == self.jobs[1]["name"]:
                self.assertEqual(record["status"], "failed")
                raise KeyboardInterrupt(interruption)
            return record

        with ExitStack() as patches:
            replacements = (
                (protocol, "load_config", {"return_value": (self.config, self.inventory, archive / "cluster")}),
                (protocol, "reference_index_path", {"return_value": archive / "reference/campaign.json"}),
                (protocol, "historical_preflight_path", {"return_value": archive / "reference/preflight.json"}),
                (protocol, "frozen_workloads", {"return_value": archive / "workload"}),
                (inference, "_experiment_inputs", {"return_value": {
                    key: archive / key for key in self.hashes
                    if key.startswith(("workload/", "infraestructura/"))}}),
                (manifest, "generate_run_id", {"side_effect": ["id", *sources]}),
                (manifest, "capture_git", {"return_value": self.git}),
                (manifest, "capture_nvidia", {"side_effect": AssertionError("consulta GPU real")}),
                (manifest, "utc_now", {"side_effect": lambda: stamp(now)}),
                (runner, "time", {"new": clock}),
                (inference, "time", {"new": clock}),
                (runner, "reservation", {"return_value": nullcontext()}),
                (runner.runtime, "remote", {"side_effect": AssertionError("consulta remota real")}),
                (runner.runtime, "preflight", {"return_value": self.preflight}),
                (inference, "run_job", {"side_effect": install_sealed_run}),
                (runner, "_finish_attempt", {"side_effect": finish_then_interrupt}),
            )
            for module, attribute, options in replacements:
                patches.enter_context(mock.patch.object(module, attribute, **options))
            directory = runner.run_campaign(archive / "protocol", results_root=output)

        index = inference.read_json(directory / "campaign.json")
        self.assertIn(interruption, " ".join(index["errors"]))
        self.assertEqual([entry["job"] for entry in index["runs"]], [self.jobs[0]["name"]])
        self.assertEqual([attempt["status"] for attempt in index["attempts"]],
                         ["sealed", "verification_pending"])
        data = report.build_report(directory)
        self.assertEqual(data["status"], "partial")
        self.assertEqual(len(data["runs"]), 1)
        self.assertEqual(len(data["cells"]), 7)
        self.assertEqual({cell["run_id"] for cell in data["cells"]}, {first_path.name})
        self.assertEqual(data["pending_jobs"], [job["name"] for job in self.jobs[2:]])
        self.assertEqual(len(data["pending_jobs"]), 16)
        self.assertEqual(data["attempts"][1]["status"], "verification_pending")
        self.assertEqual(data["attempts"][1]["manifest_status"], "failed")

    def test_complete_aggregates_all_verified_points(self):
        cluster_file = self.directory / "cluster.json"
        inference.write_new(cluster_file, self.inventory)
        self.inputs()
        self.hashes["cluster"] = manifest.sha256_file(cluster_file)
        refs = []
        for index, job in enumerate(self.jobs):
            start = 3 * index
            path = self.start_fixture_run(job, {}, job["name"], started=start)
            write_fixture_roster(path, job["model"], self.inventory["groups"][job["group"]], self.preflight)
            occupation = self.write_occupation(path, start, start + 1)
            with mock.patch.object(manifest, "utc_now", return_value=stamp(start + 2)):
                manifest.finalize_run(path)
            refs.append({"job": job["name"], "group": job["group"], "block": job["block"],
                         "model_key": job["model"]["key"], "run_dir": str(path), "status": "complete",
                         "manifest_sha256": manifest.sha256_file(path / "manifest.json"), "errors": [],
                         "occupation": occupation})
        self.seal(refs, complete=True)

        # Exercise campaign aggregation with prevalidated results. The partial
        # fixture separately verifies archive, client and telemetry contents.
        def verified_points(path, run, job, config, probes, hashes):
            cells = []
            for coordinate in job["cells"]:
                rate = coordinate["rate_rps"]
                model_key = job["model"]["key"]
                digest = hashlib.sha256(f"{path.name}/{rate}".encode()).hexdigest()
                cells.append({
                    "model_key": model_key, "group": job["group"], "block": job["block"], "rate_rps": rate,
                    "receipts_sha256": {"measured": digest},
                    "summary": {
                        "phase": "measured", "model_key": model_key, "rate_rps": rate,
                        "offered": config["measurement"]["measured_requests"],
                        "slo_status": "fail" if rate == 2 else "pass", "limitations": [],
                    },
                })
            return cells, [], {}

        with (mock.patch.object(report, "_archived_inputs", return_value=({}, {})),
              mock.patch.object(report, "_verified_run", side_effect=verified_points)):
            data = report.build_report(self.directory)

        self.assertEqual(data["status"], "complete")
        self.assertEqual(len(data["runs"]), 18)
        self.assertEqual(len(data["cells"]), 126)
        self.assertEqual(data["missing_cells"], [])
        self.assertEqual(data["excluded_cells"], [])
        self.assertEqual(len(data["frontiers"]), 6)
        for frontier in data["frontiers"]:
            self.assertEqual(frontier["status"], "bracketed")
            self.assertEqual(frontier["highest_passing_rate_rps"], 1.5)
            self.assertEqual(frontier["first_higher_failing_rate_rps"], 2)

    def test_rejects_campaign_complete_without_126_verified_points(self):
        first = self.run_fixture(complete=True)
        self.seal([first], complete=True)
        with self.assertRaisesRegex(ValueError, "126 puntos"):
            report.build_report(self.directory)


if __name__ == "__main__":
    unittest.main()
