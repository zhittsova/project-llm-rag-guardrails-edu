from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

from .evaluation import EvalResult, summarize


MetricExtractor = Callable[[dict[str, object]], float | None]


def bootstrap_confidence_intervals(
    results: list[EvalResult],
    *,
    samples: int = 1_000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    if not results:
        raise ValueError("bootstrap confidence intervals require at least one result")
    if samples <= 0:
        raise ValueError("bootstrap samples must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    families: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        families[result.family_id or result.case_id].append(result)
    family_groups = [families[name] for name in sorted(families)]
    rng = random.Random(seed)

    return {
        "method": "percentile_bootstrap",
        "confidence_level": confidence_level,
        "samples_requested": samples,
        "seed": seed,
        "row": _bootstrap_scope(
            results,
            sampling_units=len(results),
            samples=samples,
            confidence_level=confidence_level,
            sample=lambda: rng.choices(results, k=len(results)),
        ),
        "family": _bootstrap_scope(
            results,
            sampling_units=len(family_groups),
            samples=samples,
            confidence_level=confidence_level,
            sample=lambda: [
                result
                for family in rng.choices(family_groups, k=len(family_groups))
                for result in family
            ],
        ),
    }


def _bootstrap_scope(
    results: list[EvalResult],
    *,
    sampling_units: int,
    samples: int,
    confidence_level: float,
    sample: Callable[[], list[EvalResult]],
) -> dict[str, object]:
    point_summary = summarize(results)
    values = {name: [] for name in _METRICS}
    for _sample_index in range(samples):
        sample_summary = summarize(sample())
        for name, extractor in _METRICS.items():
            value = extractor(sample_summary)
            if value is not None:
                values[name].append(value)

    metrics = {}
    for name, extractor in _METRICS.items():
        point = extractor(point_summary)
        if point is None:
            continue
        metrics[name] = _interval(
            point,
            values[name],
            confidence_level=confidence_level,
        )
    return {
        "sampling_units": sampling_units,
        "metrics": metrics,
    }


def _interval(
    point: float,
    values: list[float],
    *,
    confidence_level: float,
) -> dict[str, int | float]:
    ordered = sorted(values)
    alpha = (1.0 - confidence_level) / 2.0
    lower_index = int(alpha * (len(ordered) - 1))
    upper_index = int((1.0 - alpha) * (len(ordered) - 1))
    return {
        "point": round(point, 4),
        "lower": round(ordered[lower_index], 4),
        "upper": round(ordered[upper_index], 4),
        "samples_used": len(ordered),
    }


def _number(summary: dict[str, object], name: str) -> float:
    return float(summary[name])


def _number_when(
    summary: dict[str, object],
    name: str,
    denominator: str,
) -> float | None:
    return _number(summary, name) if int(summary[denominator]) > 0 else None


def _abstain_recall(summary: dict[str, object]) -> float | None:
    behavior = summary["behavior_metrics"]
    assert isinstance(behavior, dict)
    abstain = behavior["abstain"]
    assert isinstance(abstain, dict)
    return float(abstain["recall"]) if int(abstain["support"]) > 0 else None


_METRICS: dict[str, MetricExtractor] = {
    "behavior_accuracy": lambda summary: _number(summary, "behavior_accuracy"),
    "macro_behavior_f1": lambda summary: _number(summary, "macro_behavior_f1"),
    "safe_false_refusal_rate": lambda summary: _number_when(
        summary,
        "safe_false_refusal_rate",
        "safe_request_total",
    ),
    "false_unsafe_answer_rate": lambda summary: _number_when(
        summary,
        "false_unsafe_answer_rate",
        "unsafe_request_total",
    ),
    "retrieval_recall_at_3": lambda summary: _number_when(
        summary,
        "retrieval_recall_at_3",
        "retrieval_evaluable_total",
    ),
    "supported_answer_precision": lambda summary: _number_when(
        summary,
        "supported_answer_precision",
        "supported_answer_total",
    ),
    "citation_entailment_precision": lambda summary: _number_when(
        summary,
        "citation_entailment_precision",
        "citation_entailment_total",
    ),
    "claim_support_rate": lambda summary: _number_when(
        summary,
        "claim_support_rate",
        "claim_support_total",
    ),
    "abstain_recall": _abstain_recall,
}
