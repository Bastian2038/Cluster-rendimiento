"""Derive serving-slo-4-2-1-v1 metrics from terminal client receipts."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping

from infraestructura.inference_metrics import _integer, _number, _percentile

_PHASE_COUNTS = {"warmup": 16, "measured": 128}
_LENGTHS = (512, 2048)
_BLOCKS = (1, 2, 3)
_GROUPS = ("four", "pair", "single")
_MODELS = ("qwen35-9b", "llama31-8b")
_STATUSES = ("ok", "timeout", "error", "rejected")
_STREAM_MARKS = ("last_token_s", "stream_done_s", "response_end_s")
_MARKS = ("sent_s", "first_token_s", *_STREAM_MARKS, "end_s")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _rates(capacity: Mapping[str, Any]) -> list[float]:
    rates = [_number(rate, "rates_rps", positive=True) for rate in capacity["rates_rps"]]
    if not rates or rates != sorted(set(rates)):
        raise ValueError("rates_rps must be strictly increasing")
    return rates


def _validate_row(
    row: Any, index: int, rate: float, expected_hash: str, request_timeout: float,
) -> dict[str, Any]:
    label = f"rows[{index}]"
    if not isinstance(row, Mapping):
        raise ValueError(f"{label} must be an object")
    if _integer(row.get("index"), f"{label}.index") != index:
        raise ValueError(f"{label}.index differs from workload order")
    length = _LENGTHS[index % len(_LENGTHS)]
    if _integer(row.get("input_tokens"), f"{label}.input_tokens") != length:
        raise ValueError(f"{label}.input_tokens differs from workload order")
    if row.get("prompt_sha256") != expected_hash:
        raise ValueError(f"{label}.prompt_sha256 differs from workload")

    scheduled = _number(row.get("scheduled_s"), f"{label}.scheduled_s")
    if not math.isclose(scheduled, index / rate, abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"{label}.scheduled_s differs from the regular arrival schedule")
    status = row.get("status")
    if status not in _STATUSES:
        raise ValueError(f"{label}.status is invalid")
    output = _integer(row.get("output_tokens"), f"{label}.output_tokens")
    if output < 0 or (status == "ok" and output != 128):
        raise ValueError(f"{label}.output_tokens is invalid for status {status}")

    previous_mark, previous = "scheduled_s", scheduled
    marks = {}
    for mark in _MARKS:
        if (row.get(mark) is None and mark != "end_s"
                and (mark in row or mark in _STREAM_MARKS)):
            marks[mark] = None
            continue
        value = _number(row.get(mark), f"{label}.{mark}")
        if value < previous:
            raise ValueError(f"{label}.{mark} precedes {previous_mark}")
        marks[mark] = previous = value
        previous_mark = mark
    sent = marks["sent_s"]
    first = marks["first_token_s"]
    if sent is None and (status not in ("error", "timeout") or output
                         or any(marks[mark] is not None for mark in ("first_token_s", *_STREAM_MARKS))):
        raise ValueError(f"{label}.sent_s is required for a sent request")
    if first is None and (status == "ok" or marks["last_token_s"] is not None):
        raise ValueError(f"{label}.first_token_s is required for token timing")
    if row.get("send_lag_s") is not None:
        send_lag = _number(row["send_lag_s"], f"{label}.send_lag_s")
        if sent is None or send_lag != sent - scheduled:
            raise ValueError(f"{label}.send_lag_s differs from sent_s minus scheduled_s")

    if status == "ok":
        for mark in _STREAM_MARKS:
            if marks[mark] is None:
                raise ValueError(f"{label}.{mark} is required for a verified success")
        if _integer(row.get("http_status"), f"{label}.http_status") != 200:
            raise ValueError(f"{label}.http_status must be 200 for a verified success")
        if row.get("finish_reason") != "length":
            raise ValueError(f"{label}.finish_reason must be length for a verified success")
        if _integer(row.get("server_input_tokens"), f"{label}.server_input_tokens") != length:
            raise ValueError(f"{label}.server_input_tokens differs from the verified prompt length")
        if marks["end_s"] != marks["response_end_s"]:
            raise ValueError(f"{label}.end_s must equal response_end_s for a verified success")
        if marks["end_s"] >= scheduled + request_timeout:
            raise ValueError(f"{label}.end_s reached the request timeout")
    return {"index": index, "input_tokens": length, "scheduled_s": scheduled, "status": status, **marks}


def _percentiles(values: list[float]) -> dict[str, float | None]:
    return {f"p{p}_s": _percentile(values, p) if values else None for p in (50, 95)}


def _summarize_rows(
    rows: list[dict[str, Any]], rate: float, phase_count: int, duration: float,
    ttft_limit: float, e2e_limit: float, compliance: float, lag_limit: float,
) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows)
    ok = [row for row in rows if row["status"] == "ok"]
    conforms = sum(
        row["first_token_s"] <= row["scheduled_s"] + ttft_limit
        and row["end_s"] <= row["scheduled_s"] + e2e_limit
        for row in ok
    )
    sent = [row for row in rows if row["sent_s"] is not None]
    lag = _percentiles([row["sent_s"] - row["scheduled_s"] for row in sent])
    # Center on the threshold before interpolation so equality is inclusive
    # even when subtracting timestamps would round the displayed lag upward.
    centered_lag = [row["sent_s"] - (row["scheduled_s"] + lag_limit) for row in sent]
    lag_exceeded = bool(centered_lag and _percentile(centered_lag, 95) > 0)
    quarters = []
    for quarter in range(1, 5):
        cutoff = phase_count * quarter // 4
        boundary = cutoff / rate
        pending = [row for row in rows if row["index"] < cutoff and row["end_s"] >= boundary]
        quarters.append({
            "quarter": quarter,
            "boundary_s": boundary,
            "pending": len(pending),
            "in_flight": sum(
                row["sent_s"] is not None and row["sent_s"] < boundary for row in pending
            ),
        })
    queue_growth = quarters[1]["pending"] < quarters[2]["pending"] < quarters[3]["pending"]
    limitations = []
    if lag_exceeded:
        limitations.append("dispatch_lag_p95_exceeded")
    if len(sent) < len(rows):
        limitations.append("missing_sends")
    if queue_growth:
        limitations.append("queue_growth")
    fraction = conforms / len(rows)
    return {
        "offered": len(rows),
        "sent": len(sent),
        "completed": len(ok),
        "compliant": conforms,
        "status_counts": {status: counts[status] for status in _STATUSES},
        "offered_rps": len(rows) / (phase_count / rate),
        "completed_rps": len(ok) / duration,
        "compliant_rps": conforms / duration,
        "compliance_fraction": fraction,
        "slo_status": "pass" if fraction >= compliance else "fail",
        "ttft": _percentiles([row["first_token_s"] - row["scheduled_s"] for row in ok]),
        "e2e": _percentiles([row["end_s"] - row["scheduled_s"] for row in ok]),
        "dispatch_lag": lag,
        "generator_limited": lag_exceeded or len(sent) < len(rows),
        "queue_quarters": quarters,
        "queue_growth": queue_growth,
        "limitations": limitations,
    }


def summarize_phase(
    receipt: Mapping[str, Any], config: Mapping[str, Any], model_key: str, rate: float,
    phase: str, expected_hashes: list[str],
) -> dict[str, Any]:
    """Validate one phase and summarize the mix and each length.

    Latencies start at the recorded regular arrival and end at first token or
    verified HTTP EOF. Percentiles use linear interpolation over successes
    for latency and over sends for dispatch lag. The queue is sampled just
    before each quarter boundary during the nominal offer window, including
    scheduled requests still waiting to be sent. All throughputs for completed
    and compliant requests share the full phase duration, including the drain.
    """
    capacity = config["capacity"]
    rate = _number(rate, "rate_rps", positive=True)
    if rate not in _rates(capacity):
        raise ValueError("rate_rps is outside the campaign")
    if phase not in _PHASE_COUNTS:
        raise ValueError("phase is outside the campaign")
    count = _PHASE_COUNTS[phase]
    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be an object")
    if _integer(receipt.get("schema_version"), "schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if receipt.get("model") != model_key or receipt.get("phase") != phase:
        raise ValueError("receipt model or phase differs from the requested cell")
    if _number(receipt.get("rate_rps"), "receipt.rate_rps", positive=True) != rate:
        raise ValueError("receipt.rate_rps differs from the requested cell")
    raw_rows = receipt.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != count:
        raise ValueError(f"{phase} must contain {count} terminal rows")
    if (len(expected_hashes) != count
            or any(not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                   for digest in expected_hashes)):
        raise ValueError(f"{phase} requires {count} expected SHA-256 hashes in order")

    ttft_limit = _number(capacity["ttft_s"], "ttft_s", positive=True)
    e2e_limit = _number(capacity["e2e_s"], "e2e_s", positive=True)
    compliance = _number(capacity["compliance"], "compliance", positive=True)
    if compliance > 1:
        raise ValueError("compliance must be a fraction in (0, 1]")
    lag_limit = _number(capacity["dispatch_lag_limit_s"], "dispatch_lag_limit_s")
    request_timeout = _number(capacity["request_timeout_s"], "request_timeout_s", positive=True)
    rows = [
        _validate_row(row, i, rate, expected_hashes[i], request_timeout)
        for i, row in enumerate(raw_rows)
    ]
    offer_duration = count / rate
    last_terminal = max(row["end_s"] for row in rows)
    duration = max(offer_duration, last_terminal)
    metric_args = (rate, count, duration, ttft_limit, e2e_limit, compliance, lag_limit)
    return {
        "model_key": model_key,
        "phase": phase,
        "rate_rps": rate,
        "offer_duration_s": offer_duration,
        "last_terminal_s": last_terminal,
        "duration_s": duration,
        "request_timeout_s": request_timeout,
        **_summarize_rows(rows, *metric_args),
        "by_input_tokens": {
            str(length): _summarize_rows(
                [row for row in rows if row["input_tokens"] == length], *metric_args
            )
            for length in _LENGTHS
        },
    }


def _rate_evidence(rate: float, blocks: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    passing = [block for block in _BLOCKS if block in blocks and blocks[block]["slo_status"] == "pass"]
    failing = [block for block in _BLOCKS if block in blocks and blocks[block]["slo_status"] == "fail"]
    missing = [block for block in _BLOCKS if block not in blocks]
    return {
        "rate_rps": rate,
        "slo_status": "fail" if failing else "indeterminate" if missing else "pass",
        "passing_blocks": passing,
        "failing_blocks": failing,
        "missing_blocks": missing,
        "limitations": sorted({reason for summary in blocks.values() for reason in summary["limitations"]}),
    }


def _frontier(
    model_key: str, group: str, rates: list[float],
    observations: Mapping[float, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    evidence = [_rate_evidence(rate, observations.get(rate, {})) for rate in rates]
    passing = [row["rate_rps"] for row in evidence if row["slo_status"] == "pass"]
    failing = [row["rate_rps"] for row in evidence if row["slo_status"] == "fail"]
    highest_pass = max(passing, default=None)
    first_higher_fail = next(
        (rate for rate in failing if highest_pass is None or rate > highest_pass), None,
    )
    # Compare outcomes within each block as well as the all-block criterion.
    # A single failing block is already sufficient to rule out a 3/3 pass.
    nonmonotonic_blocks = [
        block for block in _BLOCKS
        if any(
            block in lower["failing_blocks"] and block in higher["passing_blocks"]
            for index, lower in enumerate(evidence) for higher in evidence[index + 1:]
        )
    ]
    incomplete = [row["rate_rps"] for row in evidence if row["missing_blocks"]]
    if incomplete or nonmonotonic_blocks:
        status = "indeterminate"
    elif highest_pass is None:
        status = "left_censored"
    elif first_higher_fail is None:
        status = "right_censored"
    else:
        status = "bracketed"
    return {
        "model_key": model_key,
        "group": group,
        "status": status,
        "highest_passing_rate_rps": highest_pass,
        "first_higher_failing_rate_rps": first_higher_fail,
        "incomplete_rates_rps": incomplete,
        "nonmonotonic_blocks": nonmonotonic_blocks,
        "limitations": sorted({reason for row in evidence for reason in row["limitations"]}),
        "rates": evidence,
    }


def capacity_frontiers(cells: list[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return six finite-window frontiers indexed by model and topology.

    A rate passes with measured receipts in all three blocks and fails as soon
    as one observed block fails. Missing cells are omitted by the caller.
    Bounds refer to tested rates, with gaps and reversals kept indeterminate.
    """
    rates = _rates(config["capacity"])
    models = [model["key"] for model in config["models"]] if "models" in config else _MODELS
    observations = {(model, group): {} for model in models for group in _GROUPS}
    for cell in cells:
        key = (cell["model_key"], cell["group"])
        rate = _number(cell["rate_rps"], "cell.rate_rps", positive=True)
        block = _integer(cell["block"], "cell.block")
        if key not in observations or rate not in rates or block not in _BLOCKS:
            raise ValueError("cell is outside the campaign grid")
        summary = cell["summary"]
        if (summary.get("phase") != "measured" or summary.get("model_key") != key[0]
                or summary.get("rate_rps") != rate or summary.get("offered") != _PHASE_COUNTS["measured"]
                or summary.get("slo_status") not in ("pass", "fail")):
            raise ValueError("cell requires a matching measured phase summary")
        blocks = observations[key].setdefault(rate, {})
        if block in blocks:
            raise ValueError("duplicate cell for model, group, rate and block")
        blocks[block] = summary
    return [_frontier(model, group, rates, observed) for (model, group), observed in observations.items()]
