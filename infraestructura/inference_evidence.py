"""Local verification of archived workload and client evidence for inference 4/2/1."""

from __future__ import annotations

import json
from datetime import datetime
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
# Keep the historical source contract frozen. New protocols add their own inputs.
DIAGNOSIS_PROTOCOL = 'inference-diagnosis-2048-c8-v1'
DIAGNOSTIC_SOURCES = (
    'infraestructura/inference_diag_protocol.py',
    'infraestructura/inference_diag_phases.py',
    'infraestructura/inference_diag_client.py',
    'infraestructura/inference_diag_traces.py',
    'infraestructura/inference_diag_quality.py',
)
SOURCES_BY_PROTOCOL = {
    'inference-4-2-v1': RUNNER_SOURCES,
    'inference-4-2-1-v1': RUNNER_SOURCES,
    DIAGNOSIS_PROTOCOL: RUNNER_SOURCES + DIAGNOSTIC_SOURCES,
}


def runner_sources(protocol_id):
    return SOURCES_BY_PROTOCOL[protocol_id]


def cell_id(cell):
    name = f"{cell['input_tokens']}-c{cell['concurrency']}"
    return name + ('-' + cell['mode'] if 'mode' in cell else '')


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
    protocol = run['requested_config']['protocol']
    code = _archived_hashes(path, inputs, runner_sources(protocol['protocol_id']), shared_issues)
    if protocol['protocol_id'] == DIAGNOSIS_PROTOCOL:
        _archived_hashes(path, inputs, ('cluster', 'protocol', 'diagnosis/historical-campaign.json'), shared_issues)
        try:
            from infraestructura import inference_diag_protocol
            inference_diag_protocol.validate_archived_historical(
                protocol, manifest._internal_file(path, 'raw/inputs/diagnosis/historical-campaign.json'))
            archived_protocol = json.loads((path / 'raw/inputs/protocol').read_text(encoding='utf-8'))
            if archived_protocol != protocol:
                raise ValueError('archived protocol differs from requested configuration')
        except (OSError, ValueError, KeyError) as exc:
            shared_issues.append(f'diagnosis input pin: {exc}')
    model = job["model"]
    workloads, issues_by_length = {}, {}
    for length in (512, 2048):
        issues = shared_issues.copy()
        prefix = f"workload/{model['key']}/{length}"
        names = [f"{prefix}/{filename}" for filename in _PROFILE_FILES]
        hashes = _archived_hashes(path, inputs, names, issues)
        if protocol['protocol_id'] == DIAGNOSIS_PROTOCOL:
            pinned = protocol['diagnosis']['historical_jsonl_sha256']
            for phase in _PHASES:
                name = f'{prefix}/{phase}.jsonl'
                if hashes.get(name) != pinned.get(name):
                    issues.append(f'{name}: differs from pinned historical JSONL')
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
        relative = f"raw/nodes/pegasus/metrics/{cell_id(cell)}/{phase}.json"
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


def _verify_profile_control(control):
    """Validate observed START/STOP acknowledgements independently of Kineto.

    Raise ValueError for an invalid window. The context's exit can follow STOP:
    a bounded profile may cover only a prefix of the measured client batch.
    """
    if control.get('valid') is not True:
        raise ValueError('profiler control invalid')
    stamps = []
    for action in ('start', 'stop'):
        command = control[action]
        if (command.get('http_status') not in (200, 204) or command.get('error')
                or command.get('method') != 'POST' or command.get('path') != f'/{action}_profile'):
            raise ValueError(f'{action} acknowledgement missing')
        for mark in ('sent', 'ack'):
            stamp = command[mark]
            monotonic = stamp['monotonic_ns']
            if (type(monotonic) is not int or monotonic <= 0
                    or datetime.fromisoformat(stamp['utc']).tzinfo is None):
                raise ValueError(f'{action}.{mark} requires observed clock marks')
            stamps.append(monotonic)
    if stamps != sorted(stamps):
        raise ValueError('profiler START/STOP clock marks out of order')
    elapsed = (stamps[-1] - stamps[0]) / 1e9
    if not 0 <= elapsed <= 60:
        raise ValueError('profiler control exceeds the approved window')


def diagnostic_cell_issues(path, model, cell, ranks):
    """Recompute instruments using only sealed evidence, preserving basic metrics."""
    from infraestructura import inference_diag_phases as phases
    from infraestructura import inference_diag_client as client
    from infraestructura import inference_diag_traces as traces
    path = Path(path)
    directory = path / 'raw/diagnostic' / cell_id(cell)
    def read(name):
        return json.loads((directory / name).read_text(encoding='utf-8'))
    issues = []
    raw_dir = path / 'raw/nodes/pegasus/metrics' / cell_id(cell)
    measured = None
    try:
        measured = json.loads((raw_dir / 'measured.json').read_text(encoding='utf-8'))
        delta = phases.phase_deltas(read('before.json'), read('after.json'), model['key'], expected_count=32,
                                    measured=measured)
        if delta.get('valid') is not True or delta != read('phases.json'):
            raise ValueError('phase evidence is invalid or receipt differs from recomputation')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        issues.append(f'phases: {exc}')
    if 'mode' in cell:
        try:
            events = json.loads((raw_dir / 'client-events.json').read_text(encoding='utf-8'))
            measured = json.loads((raw_dir / 'measured.json').read_text(encoding='utf-8'))
            checked = client.verify_events(events, measured)
            if checked.get('valid') is not True or checked != read('client-validation.json'):
                raise ValueError('client events are invalid or validation receipt differs')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append(f'client events: {exc}')
    if cell.get('mode') == 'on':
        try:
            _verify_profile_control(read('profile-control.json'))
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError) as exc:
            issues.append(f'profile control: {exc}')
        try:
            control = read('profile-control.json')
            events = json.loads((raw_dir / 'client-events.json').read_text(encoding='utf-8'))
            measured = json.loads((raw_dir / 'measured.json').read_text(encoding='utf-8'))
            coverage = client.profile_coverage(control, events, measured)
            if coverage.get('valid') is not True or coverage.get('count', 0) < 1 or coverage != read('profile-coverage.json'):
                raise ValueError('measured requests do not overlap profile or coverage receipt differs')
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError) as exc:
            issues.append(f'profile coverage: {exc}')
        # Rehash/identity verification must run even with an invalid control
        # receipt, and a missing rank file must not invalidate good acuses.
        try:
            receipt = read('traces.json')
            checked = traces.validate_traces(directory / 'traces', receipt, ranks)
            if checked.get('valid') is not True or checked != receipt:
                raise ValueError('traces invalid or receipt differs from recomputation')
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            issues.append(f'traces: {exc}')
    return issues
