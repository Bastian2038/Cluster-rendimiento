"""Local verification of archived workload and client evidence for inference 4/2/1."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from infraestructura import manifest
from infraestructura.inference_metrics import (
    load_bench_result, summarize_bench_result, validate_client_identity,
)


RUNNER_SOURCES = (
    "infraestructura/__init__.py",
    "infraestructura/inference.py",
    "infraestructura/inference_runtime.py",
    "infraestructura/inference_worker.py",
    "infraestructura/inference_workload.py",
    "infraestructura/inference_metrics.py",
    "infraestructura/inference_observation.py",
    "infraestructura/inference_evidence.py",
    "infraestructura/cluster.py",
    "infraestructura/manifest.py",
    "infraestructura/rutas.py",
)
_GENERATOR = "infraestructura/inference_workload.py"
_PHASES = ("warmup", "measured")
_PROFILE_FILES = ("warmup.jsonl", "measured.jsonl", "metadata.json", "reader-validation.json")


def _archived_hashes(path, inputs, names, issues):
    hashes = {}
    for name in names:
        records = inputs.get(name, [])
        if len(records) != 1:
            issues.append(f"{name}: expected one input record, found {len(records)}")
            continue
        try:
            archived = manifest._internal_file(path, f"raw/inputs/{name}")
            digest = manifest.sha256_file(archived)
            if digest != records[0].get("sha256"):
                raise ValueError("archived SHA-256 differs from input record")
        except (OSError, ValueError, manifest.ManifestError) as exc:
            issues.append(f"{name}: {exc}")
        else:
            hashes[name] = digest
    return hashes


def _verify_profile(path, prefix, length, model, protocol, generator_hash, hashes):
    """Bind the generator and the CustomDataset receipt to this exact input profile."""
    directory = path / "raw/inputs" / prefix
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    reader = json.loads((directory / "reader-validation.json").read_text(encoding="utf-8"))
    expected = {
        "model_id": model["id"], "model_revision": model["revision"],
        "tokenizer_id": model["tokenizer_id"], "tokenizer_revision": model["tokenizer_revision"],
        "input_tokens": length, "output_tokens": protocol["workload"]["output_tokens"],
        "seed": protocol["workload"]["seed"], "generator_sha256": generator_hash,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f"metadata.json: {key} differs from the requested profile")
    if (reader["status"] != "validated" or reader["seed"] != expected["seed"]
            or reader["reader"] != "vllm.benchmarks.datasets.datasets.CustomDataset"):
        raise ValueError("reader-validation.json: expected validation by the configured reader")
    for phase in _PHASES:
        digest = hashes[f"{prefix}/{phase}.jsonl"]
        receipt = reader["phases"][phase]
        if metadata["files_sha256"][phase] != digest or receipt["jsonl_sha256"] != digest:
            raise ValueError(f"{phase}.jsonl: SHA-256 differs from metadata or reader validation")
        count = protocol["measurement"][f"{phase}_requests"]
        if type(receipt["count"]) is not int or receipt["count"] != count:
            raise ValueError(f"reader-validation.json: {phase} count differs from the protocol")
        samples = receipt["samples"]
        if (not isinstance(samples, list) or len(samples) != count
                or any(sample["prompt_len"] != length or sample["expected_output_len"] != expected["output_tokens"]
                       for sample in samples)):
            raise ValueError(f"reader-validation.json: {phase} samples differ from the requested profile")


def archived_job_evidence(path: str | Path, run: dict, job: dict) -> tuple[
    dict[str, str], dict[int, dict[str, str]], dict[int, list[str]],
]:
    """Verify the runner sources and the workload files for each selected profile.

    Return ``(code_hashes, workload_hashes_by_length, issues_by_length)``. Hashes
    map full logical input names to SHA-256 of their verified archived copies.
    The two per-length dicts have integer keys 512 and 2048. Common source
    problems apply to both profiles. An empty issue list qualifies that profile
    for further checks. Only the run archive is read, never the live input paths.
    """
    path = Path(path)
    inputs = defaultdict(list)
    records = run.get("inputs")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and isinstance(record.get("name"), str):
                inputs[record["name"]].append(record)
    shared_issues = []
    code = _archived_hashes(path, inputs, RUNNER_SOURCES, shared_issues)
    model = job["model"]
    workloads, issues_by_length = {}, {}
    for length in (512, 2048):
        issues = shared_issues.copy()
        prefix = f"workload/{model['key']}/{length}"
        names = [f"{prefix}/{filename}" for filename in _PROFILE_FILES]
        hashes = _archived_hashes(path, inputs, names, issues)
        if len(hashes) == len(names) and _GENERATOR in code:
            try:
                _verify_profile(path, prefix, length, model, run["requested_config"]["protocol"],
                                code[_GENERATOR], hashes)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                issues.append(f"{prefix}: {exc}")
        workloads[length] = hashes
        issues_by_length[length] = issues
    return code, workloads, issues_by_length


def verify_client_cell(path: str | Path, model: dict, cell: dict, protocol: dict) -> list[str]:
    """Verify both original Pegasus client results and reproduce the published summary.

    The caller verifies archive integrity. This checks the requested work and
    client identity of each phase, and compares all measured summary fields
    (including T32, per-trial percentiles and counts) before report rounding.
    Each issue belongs only to this cell. Other cells remain independently usable.
    """
    length, concurrency = cell.get("input_tokens"), cell.get("concurrency")
    workload = protocol["workload"]
    if type(length) is not int or length not in workload["input_tokens"]:
        return ["input_tokens is outside the requested workload"]
    if type(concurrency) is not int or concurrency not in workload["concurrencies"]:
        return ["concurrency is outside the requested workload"]

    issues = []
    for phase in _PHASES:
        relative = f"raw/nodes/pegasus/metrics/{length}-c{concurrency}/{phase}.json"
        try:
            raw = load_bench_result(manifest._internal_file(Path(path), relative))
            validate_client_identity(raw, model=model, concurrency=concurrency)
            summary = summarize_bench_result(
                raw, expected_prompt_tokens=length, expected_output_tokens=workload["output_tokens"],
                expected_count=protocol["measurement"][f"{phase}_requests"],
            )
            if phase == "measured":
                # JSON equality preserves types of counts (False == 0 in Python).
                changed = [key for key, value in summary.items() if key not in cell or
                           json.dumps(cell[key], sort_keys=True) != json.dumps(value, sort_keys=True)]
                if changed:
                    raise ValueError("measured summary differs in " + ", ".join(changed))
        except (OSError, ValueError, TypeError, KeyError, OverflowError, manifest.ManifestError) as exc:
            issues.append(f"{relative}: {exc}")
    return issues
