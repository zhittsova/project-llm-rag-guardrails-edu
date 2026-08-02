from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from guardrails_llm.inhouse_experiment import (
    build_balanced_classifier_benchmark,
    derive_classifier_label,
)
from guardrails_llm.model_calibration import ClassifierPrediction
from guardrails_llm.model_config import OpenAIModelConfig
from guardrails_llm.qwen3guard import QWEN3GUARD_MODEL, Qwen3GuardResult
from guardrails_llm.qwen3guard_experiment import (
    Qwen3GuardPrediction,
    compare_qwen_classifier_captures,
    evaluate_qwen3guard_predictions,
    run_qwen3guard_capture,
)


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data" / "eval_cases_milestone3_v2_development.jsonl"
CALIBRATION = ROOT / "data" / "eval_cases_milestone3_v2_calibration.jsonl"


class FakeQwen3GuardClassifier:
    model_name = QWEN3GUARD_MODEL

    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> Qwen3GuardResult:
        self.calls.append(text)
        return Qwen3GuardResult(
            severity="unsafe",
            categories=("jailbreak",),
            raw_text="Safety: Unsafe\nCategories: Jailbreak",
        )


def _configure_inhouse(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )


def _allowed_config() -> OpenAIModelConfig:
    return OpenAIModelConfig(
        classifier_model=QWEN3GUARD_MODEL,
        allow_remote_models=True,
    )


def _native_prediction_for(case) -> Qwen3GuardPrediction:
    expected = derive_classifier_label(case)
    if expected == "safe":
        severity = "safe"
        categories: tuple[str, ...] = ()
    elif expected == "prompt_injection":
        severity = "unsafe"
        categories = ("jailbreak",)
    elif expected == "pii":
        severity = "unsafe"
        categories = ("pii",)
    elif expected == "unsafe_request":
        severity = "unsafe"
        categories = ("violent",)
    else:
        severity = "unsafe"
        categories = ("unethical acts",)
    return Qwen3GuardPrediction(
        case_id=case.case_id,
        severity=severity,
        categories=categories,
        model=QWEN3GUARD_MODEL,
        raw_text=f"Safety: {severity.title()}",
    )


def _write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "".join(json.dumps(asdict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_intervention_metrics_use_all_six_labels() -> None:
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)
    predictions = [_native_prediction_for(case) for case in cases]

    report = evaluate_qwen3guard_predictions(cases, predictions)

    assert report["intervention"]["total"] == 600
    assert report["intervention"]["accuracy"] == 1.0
    assert report["intervention"]["f1"] == 1.0
    assert report["taxonomy"]["supported_expected_cases"] == 400
    assert report["taxonomy"]["unsupported_expected_cases"] == 200
    assert report["taxonomy"]["supported_exact_accuracy"] == 1.0
    assert report["taxonomy"]["per_project_label"]["academic_integrity"][
        "outside_native_taxonomy"
    ] is True
    assert report["quality_gates"] == {
        "structured_validity": 1.0,
        "structured_validity_passed": True,
        "intervention_recall": 1.0,
        "intervention_recall_passed": True,
        "safe_false_positive_rate": 0.0,
        "safe_false_positive_rate_passed": True,
        "all_passed": True,
    }


def test_invalid_and_missing_predictions_reduce_validity_and_accuracy() -> None:
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)[:3]
    predictions = [
        Qwen3GuardPrediction(
            case_id=cases[0].case_id,
            severity=None,
            error="ValueError: malformed response",
        ),
        _native_prediction_for(cases[1]),
    ]

    report = evaluate_qwen3guard_predictions(cases, predictions)

    assert report["structured_validity"] == 1 / 3
    assert report["intervention"]["correct"] == 1
    assert report["intervention"]["accuracy"] == 1 / 3
    assert report["invalid_predictions"] == 1
    assert report["missing_predictions"] == 1


def test_capture_resumes_and_manifest_excludes_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "manifest.json"
    classifier = FakeQwen3GuardClassifier()

    first = run_qwen3guard_capture(
        config=_allowed_config(),
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        output_path=output,
        manifest_path=manifest,
        classifier=classifier,
        limit_cases=3,
        captured_at="2026-07-22T12:00:00Z",
    )
    resumed_classifier = FakeQwen3GuardClassifier()
    second = run_qwen3guard_capture(
        config=_allowed_config(),
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        output_path=output,
        manifest_path=manifest,
        classifier=resumed_classifier,
        limit_cases=3,
        captured_at="2026-07-22T12:05:00Z",
    )

    assert len(classifier.calls) == 3
    assert resumed_classifier.calls == []
    assert first["completed_cases"] == 3
    assert second["resumed_cases"] == 3
    assert second["endpoint_host"] == (
        "learning-services4.fokus.fraunhofer.de"
    )
    assert second["parser_version"] == "qwen3guard-native-v1"
    assert len(second["configuration_fingerprint"]) == 64
    serialized = manifest.read_text(encoding="utf-8")
    assert "fixture-key" not in serialized
    assert "https://" not in serialized
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_capture_rejects_resume_with_a_different_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    kwargs = {
        "config": _allowed_config(),
        "development_cases_path": DEVELOPMENT,
        "calibration_cases_path": CALIBRATION,
        "output_path": tmp_path / "predictions.jsonl",
        "manifest_path": tmp_path / "manifest.json",
        "classifier": FakeQwen3GuardClassifier(),
    }
    run_qwen3guard_capture(**kwargs, limit_cases=2)

    with pytest.raises(ValueError, match="configuration does not match"):
        run_qwen3guard_capture(**kwargs, limit_cases=3)


def test_comparison_reports_models_on_identical_cases(tmp_path: Path) -> None:
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)
    qwen = [
        ClassifierPrediction(
            case_id=case.case_id,
            predicted_label=derive_classifier_label(case),
            confidence=0.99,
        )
        for case in cases
    ]
    qwen3guard = [_native_prediction_for(case) for case in cases]

    report = compare_qwen_classifier_captures(
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        qwen_predictions_path=_write_jsonl(tmp_path / "qwen.jsonl", qwen),
        qwen3guard_predictions_path=_write_jsonl(
            tmp_path / "qwen3guard.jsonl",
            qwen3guard,
        ),
    )

    assert report["case_alignment"]["identical_case_ids"] is True
    assert report["case_alignment"]["total"] == 600
    assert report["qwen"]["intervention"]["total"] == 600
    assert report["qwen"]["project_label_accuracy"] == 1.0
    assert report["qwen3guard"]["intervention"]["total"] == 600
    assert report["qwen3guard"]["taxonomy"][
        "unsupported_expected_cases"
    ] == 200


def test_comparison_rejects_incomplete_case_alignment(tmp_path: Path) -> None:
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)
    qwen = [
        ClassifierPrediction(
            case_id=case.case_id,
            predicted_label=derive_classifier_label(case),
            confidence=0.99,
        )
        for case in cases
    ]
    qwen3guard = [_native_prediction_for(case) for case in cases[:-1]]

    with pytest.raises(ValueError, match="identical complete case IDs"):
        compare_qwen_classifier_captures(
            development_cases_path=DEVELOPMENT,
            calibration_cases_path=CALIBRATION,
            qwen_predictions_path=_write_jsonl(
                tmp_path / "qwen.jsonl",
                qwen,
            ),
            qwen3guard_predictions_path=_write_jsonl(
                tmp_path / "qwen3guard.jsonl",
                qwen3guard,
            ),
        )
