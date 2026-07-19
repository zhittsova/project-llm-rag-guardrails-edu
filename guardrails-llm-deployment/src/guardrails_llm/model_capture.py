from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

from .evaluation import EvalCase, EvalResult, load_eval_cases
from .model_calibration import (
    DEFAULT_CALIBRATION_SOURCE_CASES,
    DEFAULT_CALIBRATION_SOURCE_RESULTS,
    DEFAULT_CLASSIFIER_CALIBRATION_CASES,
    DEFAULT_JUDGE_CALIBRATION_CASES,
    ClassifierCalibrationCase,
    ClassifierPrediction,
    JudgeCalibrationCase,
    JudgePrediction,
    load_classifier_calibration_cases,
    load_judge_calibration_cases,
    validate_classifier_calibration_sources,
    validate_judge_calibration_sources,
)
from .model_config import (
    OpenAIModelConfig,
    ensure_openai_api_key,
    ensure_remote_models_allowed,
    openai_request_policy,
    resolve_openai_base_url,
)
from .openai_models import GUARD_CLASSIFIER_PROMPT_VERSION, JUDGE_PROMPT_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIER_CAPTURE_OUTPUT = (
    PROJECT_ROOT / "reports" / "model_classifier_live_predictions.jsonl"
)
DEFAULT_JUDGE_CAPTURE_OUTPUT = (
    PROJECT_ROOT / "reports" / "judge_live_predictions.jsonl"
)
DEFAULT_CAPTURE_MANIFEST_OUTPUT = (
    PROJECT_ROOT / "reports" / "model_calibration_capture_manifest.json"
)
CAPTURE_COMPONENTS = frozenset({"classifier", "judge", "both"})
CAPTURE_SELECTION_STRATEGIES = frozenset({"head", "stratified"})


def run_model_calibration_capture(
    *,
    component: str,
    config: OpenAIModelConfig,
    classifier_cases_path: Path = DEFAULT_CLASSIFIER_CALIBRATION_CASES,
    judge_cases_path: Path = DEFAULT_JUDGE_CALIBRATION_CASES,
    source_cases_path: Path = DEFAULT_CALIBRATION_SOURCE_CASES,
    source_results_path: Path = DEFAULT_CALIBRATION_SOURCE_RESULTS,
    classifier_output_path: Path = DEFAULT_CLASSIFIER_CAPTURE_OUTPUT,
    judge_output_path: Path = DEFAULT_JUDGE_CAPTURE_OUTPUT,
    manifest_output_path: Path = DEFAULT_CAPTURE_MANIFEST_OUTPUT,
    limit_cases: int | None = None,
    selection_strategy: str = "stratified",
    classifier=None,
    judge=None,
    clock: Callable[[], float] = perf_counter,
    captured_at: str | None = None,
) -> dict[str, object]:
    if component not in CAPTURE_COMPONENTS:
        raise ValueError("component must be 'classifier', 'judge', or 'both'")
    if limit_cases is not None and limit_cases < 0:
        raise ValueError("limit_cases must be zero or greater")
    if selection_strategy not in CAPTURE_SELECTION_STRATEGIES:
        raise ValueError("selection_strategy must be 'head' or 'stratified'")

    ensure_remote_models_allowed(config)
    ensure_openai_api_key(config)
    source_cases = load_eval_cases(source_cases_path)
    provider, endpoint_category = _provider_metadata(config)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "evidence_scope": "live_remote_model_capture",
        "captured_at": captured_at or _utc_now(),
        "component": component,
        "provider": provider,
        "endpoint_category": endpoint_category,
        "limit_cases": limit_cases,
        "selection_strategy": selection_strategy,
        "request_policy": openai_request_policy(config),
        "prompt_versions": {
            **(
                {"classifier": GUARD_CLASSIFIER_PROMPT_VERSION}
                if component in {"classifier", "both"}
                else {}
            ),
            **(
                {"judge": JUDGE_PROMPT_VERSION}
                if component in {"judge", "both"}
                else {}
            ),
        },
    }

    classifier_cases = []
    if component in {"classifier", "both"}:
        classifier_cases = load_classifier_calibration_cases(classifier_cases_path)
        validate_classifier_calibration_sources(classifier_cases, source_cases)
        classifier_cases = select_classifier_calibration_cases(
            classifier_cases,
            limit=limit_cases,
            strategy=selection_strategy,
        )

    judge_cases = []
    source_results: dict[str, list[dict[str, object]]] = {}
    if component in {"judge", "both"}:
        judge_cases = load_judge_calibration_cases(judge_cases_path)
        source_results = _load_source_results(source_results_path)
        validate_judge_calibration_sources(
            judge_cases,
            source_cases,
            source_results,
        )
        judge_cases = select_judge_calibration_cases(
            judge_cases,
            limit=limit_cases,
            strategy=selection_strategy,
        )

    if classifier is None and component in {"classifier", "both"}:
        from .openai_models import OpenAIGuardClassifier

        classifier = OpenAIGuardClassifier(config)
    if judge is None and component in {"judge", "both"}:
        from .openai_models import OpenAIJudge

        judge = OpenAIJudge(config)

    if component in {"classifier", "both"}:
        classifier_predictions = _capture_classifier_predictions(
            classifier_cases,
            classifier,
            provider=provider,
            clock=clock,
        )
        _write_jsonl(classifier_output_path, classifier_predictions)
        manifest["classifier"] = _capture_summary(
            classifier_predictions,
            model=classifier.model_name,
            output_path=classifier_output_path,
        )

    if component in {"judge", "both"}:
        judge_predictions = _capture_judge_predictions(
            judge_cases,
            source_cases,
            source_results,
            judge,
            provider=provider,
            clock=clock,
        )
        _write_jsonl(judge_output_path, judge_predictions)
        manifest["judge"] = _capture_summary(
            judge_predictions,
            model=judge.model_name,
            output_path=judge_output_path,
        )

    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _capture_classifier_predictions(
    cases,
    classifier,
    *,
    provider: str,
    clock: Callable[[], float],
) -> list[ClassifierPrediction]:
    predictions = []
    for case in cases:
        started_at = clock()
        predicted_label = None
        confidence = None
        explanation = None
        error = None
        try:
            result = classifier.classify(case.text)
            if result.explanation.startswith("model_classifier_error:"):
                error = result.explanation
            else:
                predicted_label = result.label
                confidence = result.confidence
                explanation = result.explanation
        except Exception as exc:
            error = f"capture_error:{type(exc).__name__}"
        latency_ms = _elapsed_ms(started_at, clock())
        try:
            prediction = ClassifierPrediction(
                case_id=case.case_id,
                predicted_label=predicted_label,
                confidence=confidence,
                error=error,
                explanation=explanation,
                provider=provider,
                model=classifier.model_name,
                latency_ms=latency_ms,
            )
        except (TypeError, ValueError) as exc:
            prediction = ClassifierPrediction(
                case_id=case.case_id,
                predicted_label=None,
                confidence=None,
                error=f"capture_error:{type(exc).__name__}",
                provider=provider,
                model=classifier.model_name,
                latency_ms=latency_ms,
            )
        predictions.append(prediction)
    return predictions


def _capture_judge_predictions(
    calibration_cases: list[JudgeCalibrationCase],
    source_cases: list[EvalCase],
    source_results: dict[str, list[dict[str, object]]],
    judge,
    *,
    provider: str,
    clock: Callable[[], float],
) -> list[JudgePrediction]:
    cases_by_id = {case.case_id: case for case in source_cases}
    results_by_scenario = {
        scenario: {
            str(payload.get("case_id")): _eval_result_from_payload(payload)
            for payload in results
        }
        for scenario, results in source_results.items()
    }
    predictions = []
    for calibration_case in calibration_cases:
        source_case = cases_by_id[calibration_case.source_case_id]
        source_result = results_by_scenario[calibration_case.scenario][
            calibration_case.source_case_id
        ]
        started_at = clock()
        result = None
        error = None
        try:
            result = judge.judge(source_case, source_result)
            error = next(
                (
                    note
                    for note in result.notes
                    if note.startswith("llm_judge_error:")
                ),
                None,
            )
        except Exception as exc:
            error = f"capture_error:{type(exc).__name__}"
        latency_ms = _elapsed_ms(started_at, clock())
        try:
            if error is None and result is not None:
                prediction = JudgePrediction(
                    case_id=calibration_case.case_id,
                    grounded=result.grounded,
                    privacy_safe=result.privacy_safe,
                    injection_safe=result.injection_safe,
                    integrity_safe=result.integrity_safe,
                    refusal_appropriate=result.refusal_appropriate,
                    score=result.score,
                    notes=result.notes,
                    provider=provider,
                    model=judge.model_name,
                    latency_ms=latency_ms,
                )
            else:
                prediction = JudgePrediction(
                    case_id=calibration_case.case_id,
                    grounded=None,
                    privacy_safe=None,
                    injection_safe=None,
                    integrity_safe=None,
                    refusal_appropriate=None,
                    score=None,
                    error=error or "capture_error:MissingResult",
                    provider=provider,
                    model=judge.model_name,
                    latency_ms=latency_ms,
                )
        except (TypeError, ValueError) as exc:
            prediction = JudgePrediction(
                case_id=calibration_case.case_id,
                grounded=None,
                privacy_safe=None,
                injection_safe=None,
                integrity_safe=None,
                refusal_appropriate=None,
                score=None,
                error=f"capture_error:{type(exc).__name__}",
                provider=provider,
                model=judge.model_name,
                latency_ms=latency_ms,
            )
        predictions.append(prediction)
    return predictions


def _capture_summary(predictions, *, model: str, output_path: Path) -> dict[str, object]:
    latencies = [float(prediction.latency_ms) for prediction in predictions]
    successful = sum(prediction.error is None for prediction in predictions)
    total = len(predictions)
    return {
        "model": model,
        "output_path": str(output_path),
        "cases_requested": total,
        "request_count": total,
        "successful_predictions": successful,
        "failed_predictions": total - successful,
        "prediction_coverage": round(successful / total, 3) if total else 0.0,
        "p50_latency_ms": round(median(latencies), 3) if latencies else None,
        "p95_latency_ms": _nearest_rank_percentile(latencies, 0.95),
    }


def _load_source_results(path: Path) -> dict[str, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(scenario, str)
        and isinstance(results, list)
        and all(isinstance(result, dict) for result in results)
        for scenario, results in payload.items()
    ):
        raise ValueError(
            f"Invalid source results at {path}: expected scenario result lists"
        )
    return payload


def _eval_result_from_payload(payload: dict[str, object]) -> EvalResult:
    allowed_fields = {field.name for field in fields(EvalResult)}
    return EvalResult(
        **{key: value for key, value in payload.items() if key in allowed_fields}
    )


def _write_jsonl(path: Path, predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(prediction)) + "\n" for prediction in predictions),
        encoding="utf-8",
    )


def _provider_metadata(config: OpenAIModelConfig) -> tuple[str, str]:
    if resolve_openai_base_url(config):
        return "openai_compatible", "custom_openai_compatible"
    return "openai", "official_openai"


def select_classifier_calibration_cases(
    cases: list[ClassifierCalibrationCase],
    *,
    limit: int | None,
    strategy: str,
) -> list[ClassifierCalibrationCase]:
    return _select_calibration_cases(
        cases,
        limit=limit,
        strategy=strategy,
        group_key=lambda case: case.expected_label,
    )


def select_judge_calibration_cases(
    cases: list[JudgeCalibrationCase],
    *,
    limit: int | None,
    strategy: str,
) -> list[JudgeCalibrationCase]:
    return _select_calibration_cases(
        cases,
        limit=limit,
        strategy=strategy,
        group_key=lambda case: (
            f"{case.expected_behavior.value}:"
            f"{case.actual_behavior is case.expected_behavior}"
        ),
    )


def _select_calibration_cases(
    cases: list,
    *,
    limit: int | None,
    strategy: str,
    group_key,
) -> list:
    if strategy not in CAPTURE_SELECTION_STRATEGIES:
        raise ValueError("selection_strategy must be 'head' or 'stratified'")
    if limit is None:
        return cases
    if limit == 0:
        return []
    if strategy == "head":
        return cases[:limit]

    buckets: dict[str, list] = {}
    for case in cases:
        buckets.setdefault(group_key(case), []).append(case)
    selected = []
    for index in range(max((len(bucket) for bucket in buckets.values()), default=0)):
        for bucket in buckets.values():
            if index < len(bucket):
                selected.append(bucket[index])
                if len(selected) == limit:
                    return selected
    return selected


def _elapsed_ms(started_at: float, finished_at: float) -> float:
    return round(max((finished_at - started_at) * 1000, 0.0), 3)


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(percentile * len(ordered)) - 1, 0)
    return round(ordered[index], 3)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
