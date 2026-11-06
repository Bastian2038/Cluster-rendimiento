"""Fail-closed parsing of vLLM 0.24.0 detailed serving benchmark JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def load_bench_result(path: str | Path) -> dict[str, Any]:
    """Read one JSON object; append-result JSONL is deliberately rejected."""
    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("benchmark result must be one JSON object")
    return result


def validate_bench_result(
    result: Mapping[str, Any], *, expected_prompt_tokens: int,
    expected_output_tokens: int = 128, expected_count: int = 32,
) -> None:
    """Reject missing/partial requests and mismatched reported token counts.

    This validates vLLM client-reported lengths, not an independent server
    tokenizer or a proof of client saturation.
    """
    if not isinstance(result, Mapping):
        raise ValueError("result must be an object")
    for name, value in (("expected_prompt_tokens", expected_prompt_tokens),
                        ("expected_output_tokens", expected_output_tokens),
                        ("expected_count", expected_count)):
        if _integer(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if result.get("backend") != "openai":
        raise ValueError("expected openai completions backend")
    if _integer(result.get("num_prompts"), "num_prompts") != expected_count:
        raise ValueError("num_prompts differs from expected count")
    if _integer(result.get("completed"), "completed") != expected_count:
        raise ValueError("benchmark has incomplete requests")
    if _integer(result.get("failed"), "failed") != 0:
        raise ValueError("benchmark reports failed requests")
    duration = _number(result.get("duration"), "duration", positive=True)
    for key, target in (("input_lens", expected_prompt_tokens),
                        ("output_lens", expected_output_tokens)):
        values = result.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f"{key} missing or wrong count; require --save-detailed")
        if any(_integer(item, key) != target for item in values):
            raise ValueError(f"{key} differs from expected {target}")
    errors = result.get("errors")
    if not isinstance(errors, list) or len(errors) != expected_count or any(errors):
        raise ValueError("errors missing or nonempty")
    for key in ("ttfts", "itls"):
        values = result.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f"{key} missing or wrong count")
        for index, item in enumerate(values):
            if key == "itls":
                if not isinstance(item, list) or not item:
                    raise ValueError(f"itls[{index}] must be a nonempty list")
                for interval in item:
                    _number(interval, f"itls[{index}]")
            else:
                _number(item, f"ttfts[{index}]", positive=True)
    if _integer(result.get("total_input_tokens"), "total_input_tokens") != expected_count * expected_prompt_tokens:
        raise ValueError("total_input_tokens inconsistent with requests")
    if _integer(result.get("total_output_tokens"), "total_output_tokens") != expected_count * expected_output_tokens:
        raise ValueError("total_output_tokens inconsistent with requests")
    for key, computed in (("request_throughput", expected_count / duration),
                          ("output_throughput", expected_count * expected_output_tokens / duration)):
        reported = _number(result.get(key), key, positive=True)
        if not math.isclose(reported, computed, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"{key} inconsistent with duration and count")


def _percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _reported_percentile(result: Mapping[str, Any], metric: str, percentage: int) -> float:
    key = f"p{percentage}_{metric}_ms"
    return _number(result.get(key), key)


def summarize_bench_result(
    result: Mapping[str, Any], *, expected_prompt_tokens: int,
    expected_output_tokens: int = 128, expected_count: int = 32,
) -> dict[str, Any]:
    """Return per-trial client metrics; duration is T32 wall time in seconds."""
    validate_bench_result(result, expected_prompt_tokens=expected_prompt_tokens,
                          expected_output_tokens=expected_output_tokens,
                          expected_count=expected_count)
    duration = float(result["duration"])
    ttfts = [float(value) * 1000 for value in result["ttfts"]]
    # vLLM defines TPOT=(E2E-TTFT)/(output_len-1); E2E per request is not
    # preserved in the detailed result, so use vLLM's reported aggregate TPOT.
    metrics = {}
    for label, values in (("ttft", ttfts),):
        metrics[label] = {f"p{p}_ms": _percentile(values, p) for p in (50, 95)}
    for label in ("tpot", "e2el"):
        metrics[label] = {f"p{p}_ms": _reported_percentile(result, label, p)
                          for p in (50, 95)}
    return {
        "duration_seconds": duration, "t32_seconds": duration if expected_count == 32 else None,
        "requests_per_second": expected_count / duration,
        "input_tokens_per_second": expected_count * expected_prompt_tokens / duration,
        "output_tokens_per_second": expected_count * expected_output_tokens / duration,
        "ttft": metrics["ttft"], "tpot": metrics["tpot"], "e2el": metrics["e2el"],
        "raw_counts": {"requested": expected_count, "completed": result["completed"],
                       "failed": result["failed"], "input_tokens": result["total_input_tokens"],
                       "output_tokens": result["total_output_tokens"]},
        "client_saturation": "undetermined: vLLM result has no client CPU/event-loop timing",
        "metric_source": "vllm-bench-serve-client-reported",
    }


def paired_speedup(two_node: Mapping[str, Any], four_node: Mapping[str, Any]) -> dict[str, float]:
    """Compute strong-scale 4/2 only for independently paired, validated trials.

    The caller must establish identical campaign/model/workload/config identity;
    this function requires the same explicit ``comparison_key`` in each input.
    """
    key = two_node.get("comparison_key")
    if not key or key != four_node.get("comparison_key"):
        raise ValueError("paired trials require the same nonempty comparison_key")
    if two_node.get("raw_counts") != four_node.get("raw_counts"):
        raise ValueError("paired trials have different work")
    t2 = _number(two_node.get("t32_seconds"), "two-node T32", positive=True)
    t4 = _number(four_node.get("t32_seconds"), "four-node T32", positive=True)
    speedup = t2 / t4
    return {"s4_2": speedup, "e4_2": speedup / 2}
