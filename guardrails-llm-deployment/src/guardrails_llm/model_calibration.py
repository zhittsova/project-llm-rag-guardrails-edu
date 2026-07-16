from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from .dispositions import ResponseDisposition
from .evaluation import DIFFICULTIES, EvalCase


CLASSIFIER_LABELS = (
    "safe",
    "prompt_injection",
    "pii",
    "academic_integrity",
    "unsafe_request",
    "unsupported",
)
JUDGE_DIMENSIONS = (
    "grounded",
    "privacy_safe",
    "injection_safe",
    "integrity_safe",
    "refusal_appropriate",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIER_CALIBRATION_CASES = (
    PROJECT_ROOT / "data" / "model_classifier_calibration_v1.jsonl"
)
DEFAULT_CLASSIFIER_CALIBRATION_PREDICTIONS = (
    PROJECT_ROOT / "tests" / "fixtures" / "classifier_predictions_v1.jsonl"
)
DEFAULT_JUDGE_CALIBRATION_CASES = (
    PROJECT_ROOT / "data" / "judge_calibration_v1.jsonl"
)
DEFAULT_JUDGE_CALIBRATION_PREDICTIONS = (
    PROJECT_ROOT / "tests" / "fixtures" / "judge_predictions_v1.jsonl"
)
DEFAULT_CALIBRATION_SOURCE_CASES = (
    PROJECT_ROOT / "data" / "eval_cases_milestone3_holdout_v3.jsonl"
)
DEFAULT_CALIBRATION_SOURCE_RESULTS = (
    PROJECT_ROOT / "reports" / "disposition_guardrail_holdout_v3_results.json"
)


@dataclass(frozen=True)
class ClassifierCalibrationCase:
    case_id: str
    source_case_id: str
    text: str
    expected_label: str
    difficulty: str
    rationale: str

    def __post_init__(self) -> None:
        _require_non_empty_strings(
            self,
            ("case_id", "source_case_id", "text", "rationale"),
        )
        if self.expected_label not in CLASSIFIER_LABELS:
            raise ValueError(
                f"{self.case_id}: expected_label must be one of "
                f"{', '.join(CLASSIFIER_LABELS)}"
            )
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(
                f"{self.case_id}: difficulty must be easy, medium, or hard"
            )


@dataclass(frozen=True)
class ClassifierPrediction:
    case_id: str
    predicted_label: str | None
    confidence: float | None
    error: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_strings(self, ("case_id",))
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError(f"{self.case_id}: error must be a non-empty string")
            if self.predicted_label is not None or self.confidence is not None:
                raise ValueError(
                    f"{self.case_id}: failed prediction must not contain a label or confidence"
                )
            return
        if self.predicted_label not in CLASSIFIER_LABELS:
            raise ValueError(
                f"{self.case_id}: predicted_label must be one of "
                f"{', '.join(CLASSIFIER_LABELS)}"
            )
        _validate_score(self.case_id, "confidence", self.confidence)


@dataclass(frozen=True)
class JudgeCalibrationCase:
    case_id: str
    source_case_id: str
    scenario: str
    expected_behavior: ResponseDisposition
    actual_behavior: ResponseDisposition
    grounded: bool
    privacy_safe: bool
    injection_safe: bool
    integrity_safe: bool
    refusal_appropriate: bool
    expected_score: float
    rationale: str

    def __post_init__(self) -> None:
        _require_non_empty_strings(
            self,
            ("case_id", "source_case_id", "scenario", "rationale"),
        )
        for field_name in ("expected_behavior", "actual_behavior"):
            try:
                behavior = ResponseDisposition(getattr(self, field_name))
            except ValueError as exc:
                raise ValueError(
                    f"{self.case_id}: {field_name} must be answer, block, abstain, or redirect"
                ) from exc
            object.__setattr__(self, field_name, behavior)
        for dimension in JUDGE_DIMENSIONS:
            if not isinstance(getattr(self, dimension), bool):
                raise ValueError(f"{self.case_id}: {dimension} must be true or false")
        _validate_score(self.case_id, "expected_score", self.expected_score)
        derived_score = round(
            sum(int(getattr(self, dimension)) for dimension in JUDGE_DIMENSIONS)
            / len(JUDGE_DIMENSIONS),
            3,
        )
        if abs(self.expected_score - derived_score) > 0.001:
            raise ValueError(
                f"{self.case_id}: expected_score must equal the mean of judge dimensions"
            )


@dataclass(frozen=True)
class JudgePrediction:
    case_id: str
    grounded: bool | None
    privacy_safe: bool | None
    injection_safe: bool | None
    integrity_safe: bool | None
    refusal_appropriate: bool | None
    score: float | None
    error: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_strings(self, ("case_id",))
        values = [getattr(self, dimension) for dimension in JUDGE_DIMENSIONS]
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError(f"{self.case_id}: error must be a non-empty string")
            if any(value is not None for value in values) or self.score is not None:
                raise ValueError(
                    f"{self.case_id}: failed prediction must not contain judge values"
                )
            return
        for dimension, value in zip(JUDGE_DIMENSIONS, values, strict=True):
            if not isinstance(value, bool):
                raise ValueError(f"{self.case_id}: {dimension} must be true or false")
        _validate_score(self.case_id, "score", self.score)


def load_classifier_calibration_cases(path: Path) -> list[ClassifierCalibrationCase]:
    return _load_jsonl(path, ClassifierCalibrationCase, "classifier calibration case")


def load_classifier_predictions(path: Path) -> list[ClassifierPrediction]:
    return _load_jsonl(path, ClassifierPrediction, "classifier prediction")


def load_judge_calibration_cases(path: Path) -> list[JudgeCalibrationCase]:
    return _load_jsonl(path, JudgeCalibrationCase, "judge calibration case")


def load_judge_predictions(path: Path) -> list[JudgePrediction]:
    return _load_jsonl(path, JudgePrediction, "judge prediction")


def validate_classifier_calibration_sources(
    cases: list[ClassifierCalibrationCase],
    source_cases: list[EvalCase],
) -> None:
    sources = {case.case_id: case for case in source_cases}
    for case in cases:
        source = sources.get(case.source_case_id)
        if source is None:
            raise ValueError(
                f"{case.case_id}: unknown source_case_id '{case.source_case_id}'"
            )
        if case.text != source.question:
            raise ValueError(f"{case.case_id}: text differs from the source question")
        if case.difficulty != source.difficulty:
            raise ValueError(f"{case.case_id}: difficulty differs from the source case")


def validate_judge_calibration_sources(
    cases: list[JudgeCalibrationCase],
    source_cases: list[EvalCase],
    source_results: dict[str, list[dict[str, object]]],
) -> None:
    sources = {case.case_id: case for case in source_cases}
    indexed_results = {
        scenario: {
            str(result.get("case_id")): result
            for result in results
            if isinstance(result, dict)
        }
        for scenario, results in source_results.items()
        if isinstance(results, list)
    }
    for case in cases:
        source = sources.get(case.source_case_id)
        if source is None:
            raise ValueError(
                f"{case.case_id}: unknown source_case_id '{case.source_case_id}'"
            )
        if source.resolved_expected_behavior() is not case.expected_behavior:
            raise ValueError(
                f"{case.case_id}: expected_behavior differs from the source case"
            )
        scenario_results = indexed_results.get(case.scenario)
        if scenario_results is None:
            raise ValueError(f"{case.case_id}: unknown scenario '{case.scenario}'")
        source_result = scenario_results.get(case.source_case_id)
        if source_result is None:
            raise ValueError(
                f"{case.case_id}: source result missing from scenario '{case.scenario}'"
            )
        if source_result.get("expected_behavior") != case.expected_behavior.value:
            raise ValueError(
                f"{case.case_id}: source result expected_behavior differs from the label"
            )
        if source_result.get("actual_behavior") != case.actual_behavior.value:
            raise ValueError(
                f"{case.case_id}: source result actual_behavior differs from the label"
            )


def evaluate_classifier_calibration(
    cases: list[ClassifierCalibrationCase],
    predictions: list[ClassifierPrediction],
) -> dict[str, object]:
    predictions_by_id = _index_predictions(cases, predictions)
    columns = CLASSIFIER_LABELS + ("parse_failure", "missing")
    confusion = {
        label: {column: 0 for column in columns}
        for label in CLASSIFIER_LABELS
    }
    details: list[dict[str, object]] = []
    valid = 0
    correct = 0
    failures = 0
    missing = 0
    for case in cases:
        prediction = predictions_by_id.get(case.case_id)
        if prediction is None:
            status = "missing"
            actual = "missing"
            missing += 1
        elif prediction.error is not None:
            status = "parse_failure"
            actual = "parse_failure"
            failures += 1
        else:
            status = "valid"
            actual = str(prediction.predicted_label)
            valid += 1
            correct += int(actual == case.expected_label)
        confusion[case.expected_label][actual] += 1
        details.append(
            {
                "case_id": case.case_id,
                "expected_label": case.expected_label,
                "predicted_label": (
                    prediction.predicted_label if prediction is not None else None
                ),
                "confidence": prediction.confidence if prediction is not None else None,
                "status": status,
                "correct": status == "valid" and actual == case.expected_label,
                "error": prediction.error if prediction is not None else None,
            }
        )

    per_label: dict[str, dict[str, int | float]] = {}
    for label in CLASSIFIER_LABELS:
        support = sum(confusion[label].values())
        predicted = sum(confusion[expected][label] for expected in CLASSIFIER_LABELS)
        true_positives = confusion[label][label]
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "support": support,
            "predicted": predicted,
            "true_positives": true_positives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    total = len(cases)
    summary = {
        "total": total,
        "predictions_received": len(predictions),
        "evaluated_predictions": valid,
        "prediction_coverage": _rate(valid, total),
        "parse_failures": failures,
        "missing_predictions": missing,
        "correct_predictions": correct,
        "accuracy_on_valid_predictions": _rate(correct, valid),
        "end_to_end_accuracy": _rate(correct, total),
        "agreement_rate": _rate(correct, total),
        "confusion_matrix": confusion,
        "per_label": per_label,
        "macro_f1": round(
            sum(float(metrics["f1"]) for metrics in per_label.values())
            / len(CLASSIFIER_LABELS),
            3,
        ),
    }
    return {"summary": summary, "results": details}


def evaluate_judge_calibration(
    cases: list[JudgeCalibrationCase],
    predictions: list[JudgePrediction],
) -> dict[str, object]:
    predictions_by_id = _index_predictions(cases, predictions)
    correct_by_dimension = {dimension: 0 for dimension in JUDGE_DIMENSIONS}
    details: list[dict[str, object]] = []
    valid = 0
    exact = 0
    failures = 0
    missing = 0
    score_errors: list[float] = []
    for case in cases:
        prediction = predictions_by_id.get(case.case_id)
        if prediction is None:
            status = "missing"
            missing += 1
            matches = {dimension: False for dimension in JUDGE_DIMENSIONS}
        elif prediction.error is not None:
            status = "parse_failure"
            failures += 1
            matches = {dimension: False for dimension in JUDGE_DIMENSIONS}
        else:
            status = "valid"
            valid += 1
            matches = {
                dimension: (
                    getattr(prediction, dimension) == getattr(case, dimension)
                )
                for dimension in JUDGE_DIMENSIONS
            }
            exact += int(all(matches.values()))
            score_errors.append(abs(float(prediction.score) - case.expected_score))
        for dimension, matches_expected in matches.items():
            correct_by_dimension[dimension] += int(matches_expected)
        details.append(
            {
                "case_id": case.case_id,
                "status": status,
                "expected_behavior": case.expected_behavior.value,
                "actual_behavior": case.actual_behavior.value,
                "exact_match": status == "valid" and all(matches.values()),
                "dimension_matches": matches,
                "expected_score": case.expected_score,
                "predicted_score": prediction.score if prediction is not None else None,
                "error": prediction.error if prediction is not None else None,
            }
        )

    total = len(cases)
    summary = {
        "total": total,
        "predictions_received": len(predictions),
        "evaluated_predictions": valid,
        "prediction_coverage": _rate(valid, total),
        "parse_failures": failures,
        "missing_predictions": missing,
        "exact_matches": exact,
        "exact_match_on_valid_predictions": _rate(exact, valid),
        "end_to_end_exact_match": _rate(exact, total),
        "dimension_accuracy": {
            dimension: _rate(correct, total)
            for dimension, correct in correct_by_dimension.items()
        },
        "dimension_accuracy_on_valid_predictions": {
            dimension: _rate(correct, valid)
            for dimension, correct in correct_by_dimension.items()
        },
        "score_mae_on_valid_predictions": (
            round(sum(score_errors) / len(score_errors), 3)
            if score_errors
            else None
        ),
    }
    return {"summary": summary, "results": details}


def run_local_model_calibration(
    *,
    classifier_cases_path: Path,
    classifier_predictions_path: Path,
    judge_cases_path: Path,
    judge_predictions_path: Path,
    source_cases_path: Path,
    source_results_path: Path,
) -> dict[str, object]:
    from .evaluation import load_eval_cases

    classifier_cases = load_classifier_calibration_cases(classifier_cases_path)
    classifier_predictions = load_classifier_predictions(
        classifier_predictions_path
    )
    judge_cases = load_judge_calibration_cases(judge_cases_path)
    judge_predictions = load_judge_predictions(judge_predictions_path)
    source_cases = load_eval_cases(source_cases_path)
    try:
        source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ValueError(
            f"Invalid source results at {source_results_path}: malformed JSON"
        ) from exc
    if not isinstance(source_results, dict):
        raise ValueError(
            f"Invalid source results at {source_results_path}: expected an object"
        )

    validate_classifier_calibration_sources(classifier_cases, source_cases)
    validate_judge_calibration_sources(judge_cases, source_cases, source_results)
    return {
        "evidence_scope": "fixture_replay_only",
        "classifier": evaluate_classifier_calibration(
            classifier_cases,
            classifier_predictions,
        ),
        "judge": evaluate_judge_calibration(judge_cases, judge_predictions),
        "limitations": [
            "Fixture predictions are not model quality evidence; they only "
            "validate calibration plumbing.",
            "Live classifier and judge outputs require a separately approved remote-model run.",
            "Human labels require disagreement review before final reporting.",
        ],
    }


def _load_jsonl(path: Path, cls, item_label: str):
    items = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid {item_label} at {path}:{line_number}: malformed JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid {item_label} at {path}:{line_number}: expected an object"
                )
            try:
                item = cls(**payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {item_label} at {path}:{line_number}: {exc}"
                ) from exc
            if item.case_id in seen_ids:
                raise ValueError(
                    f"Invalid {item_label} at {path}:{line_number}: "
                    f"duplicate case_id '{item.case_id}'"
                )
            seen_ids.add(item.case_id)
            items.append(item)
    return items


def _index_predictions(cases, predictions):
    case_ids = {case.case_id for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("calibration cases contain duplicate case_id values")
    indexed = {}
    for prediction in predictions:
        if prediction.case_id not in case_ids:
            raise ValueError(
                f"unknown prediction case_id '{prediction.case_id}'"
            )
        if prediction.case_id in indexed:
            raise ValueError(
                f"duplicate prediction case_id '{prediction.case_id}'"
            )
        indexed[prediction.case_id] = prediction
    return indexed


def _require_non_empty_strings(item: object, field_names: tuple[str, ...]) -> None:
    for field_name in field_names:
        value = getattr(item, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")


def _validate_score(case_id: str, field_name: str, value: object) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{case_id}: {field_name} must be a number from 0 to 1")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{case_id}: {field_name} must be between 0 and 1")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0
