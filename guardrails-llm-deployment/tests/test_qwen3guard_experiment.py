from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest

from guardrails_llm.inhouse_experiment import (
    build_balanced_classifier_benchmark,
    derive_classifier_label,
)
from guardrails_llm.model_calibration import ClassifierPrediction
from guardrails_llm.model_config import OpenAIModelConfig
from guardrails_llm.qwen3guard import (
    QWEN3GUARD_MODEL,
    Qwen3GuardModelUnavailableError,
    Qwen3GuardResult,
)
from guardrails_llm.qwen3guard_experiment import (
    Qwen3GuardPrediction,
    compare_qwen_classifier_captures,
    evaluate_qwen3guard_predictions,
    run_qwen3guard_capture,
)


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data" / "eval_cases_milestone3_v2_development.jsonl"
CALIBRATION = ROOT / "data" / "eval_cases_milestone3_v2_calibration.jsonl"
DATASET_MANIFEST = ROOT / "data" / "eval_cases_milestone3_v2_manifest.json"


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


class UnavailableAfterOneClassifier(FakeQwen3GuardClassifier):
    def classify(self, text: str) -> Qwen3GuardResult:
        if self.calls:
            raise RuntimeError("model is unavailable")
        return super().classify(text)


class FailingQwen3GuardClassifier(FakeQwen3GuardClassifier):
    def classify(self, text: str) -> Qwen3GuardResult:
        self.calls.append(text)
        raise ValueError("malformed response")


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


def _write_matching_manifests(tmp_path: Path) -> tuple[Path, Path]:
    baseline = json.loads(
        (ROOT / "reports" / "inhouse_classifier_v2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    qwen_manifest = tmp_path / "qwen-manifest.json"
    qwen_manifest.write_text(json.dumps(baseline), encoding="utf-8")
    qwen3guard_manifest = tmp_path / "qwen3guard-manifest.json"
    qwen3guard_manifest.write_text(
        json.dumps(
            {
                "dataset_version": baseline["dataset_version"],
                "dataset_manifest_sha256": sha256(
                    DATASET_MANIFEST.read_bytes()
                ).hexdigest(),
                "split_sha256": baseline["split_sha256"],
                "selection_sha256": baseline["selection_sha256"],
                "selected_cases": 600,
                "status": "complete",
                "completed_cases": 600,
                "failed_cases": 0,
                "model": QWEN3GUARD_MODEL,
            }
        ),
        encoding="utf-8",
    )
    return qwen_manifest, qwen3guard_manifest


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


def test_capture_stops_when_provider_model_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(Qwen3GuardModelUnavailableError, match="cannot serve"):
        run_qwen3guard_capture(
            config=_allowed_config(),
            development_cases_path=DEVELOPMENT,
            calibration_cases_path=CALIBRATION,
            output_path=output,
            manifest_path=manifest_path,
            classifier=UnavailableAfterOneClassifier(),
            limit_cases=3,
            max_concurrency=1,
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["completed_cases"] == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_capture_with_failed_rows_is_not_marked_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)

    manifest = run_qwen3guard_capture(
        config=_allowed_config(),
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        output_path=tmp_path / "predictions.jsonl",
        manifest_path=tmp_path / "manifest.json",
        classifier=FailingQwen3GuardClassifier(),
        limit_cases=2,
    )

    assert manifest["status"] == "complete_with_failures"
    assert manifest["completed_cases"] == 2
    assert manifest["failed_cases"] == 2


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
    qwen_manifest, qwen3guard_manifest = _write_matching_manifests(tmp_path)

    report = compare_qwen_classifier_captures(
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        qwen_predictions_path=_write_jsonl(tmp_path / "qwen.jsonl", qwen),
        qwen3guard_predictions_path=_write_jsonl(
            tmp_path / "qwen3guard.jsonl",
            qwen3guard,
        ),
        qwen_manifest_path=qwen_manifest,
        qwen3guard_manifest_path=qwen3guard_manifest,
    )

    assert report["comparison_scope"] == {
        "total_cases": 600,
        "comparable_cases": 400,
        "comparable_expected_labels": [
            "safe",
            "prompt_injection",
            "pii",
            "unsafe_request",
        ],
        "taxonomy_gap_cases": 200,
        "taxonomy_gap_expected_labels": [
            "academic_integrity",
            "unsupported",
        ],
        "frozen_holdout_used": False,
        "matching_capture_provenance": True,
    }
    assert report["qwen3guard"]["strict"]["total"] == 400
    assert report["qwen3guard"]["strict"]["accuracy"] == 1.0
    assert report["qwen3_baseline"]["strict"]["accuracy"] == 1.0
    assert report["taxonomy_gap"]["expected_label_counts"] == {
        "academic_integrity": 100,
        "unsupported": 100,
    }


def test_comparison_keeps_strict_and_permissive_policies_separate(
    tmp_path: Path,
) -> None:
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
    risk_case = next(
        case
        for case in cases
        if derive_classifier_label(case) == "prompt_injection"
    )
    index = next(
        index
        for index, prediction in enumerate(qwen3guard)
        if prediction.case_id == risk_case.case_id
    )
    qwen3guard[index] = Qwen3GuardPrediction(
        case_id=risk_case.case_id,
        severity="controversial",
        categories=("jailbreak",),
        model=QWEN3GUARD_MODEL,
        raw_text="Safety: Controversial\nCategories: Jailbreak",
    )
    qwen_manifest, qwen3guard_manifest = _write_matching_manifests(tmp_path)

    report = compare_qwen_classifier_captures(
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        qwen_predictions_path=_write_jsonl(tmp_path / "qwen.jsonl", qwen),
        qwen3guard_predictions_path=_write_jsonl(
            tmp_path / "qwen3guard.jsonl",
            qwen3guard,
        ),
        qwen_manifest_path=qwen_manifest,
        qwen3guard_manifest_path=qwen3guard_manifest,
    )

    assert report["qwen3guard"]["strict"]["accuracy"] == 1.0
    assert report["qwen3guard"]["permissive"]["accuracy"] == 0.9975
    assert report["qwen3guard"]["permissive"]["unsafe_recall"] == 0.9967


def test_comparison_loads_native_capture_field_names(tmp_path: Path) -> None:
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)
    qwen = [
        ClassifierPrediction(
            case_id=case.case_id,
            predicted_label=derive_classifier_label(case),
            confidence=0.99,
        )
        for case in cases
    ]
    native_rows = []
    for case in cases:
        prediction = _native_prediction_for(case)
        native_rows.append(
            {
                "case_id": prediction.case_id,
                "safety": prediction.severity.title(),
                "categories": [
                    category.title() for category in prediction.categories
                ],
                "raw_output": prediction.raw_text,
                "error": None,
                "provider": "openai_compatible",
                "model": QWEN3GUARD_MODEL,
                "latency_ms": 10.0,
            }
        )
    qwen_manifest, qwen3guard_manifest = _write_matching_manifests(tmp_path)
    native_path = tmp_path / "native.jsonl"
    native_path.write_text(
        "".join(json.dumps(row) + "\n" for row in native_rows),
        encoding="utf-8",
    )

    report = compare_qwen_classifier_captures(
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        qwen_predictions_path=_write_jsonl(tmp_path / "qwen.jsonl", qwen),
        qwen3guard_predictions_path=native_path,
        qwen_manifest_path=qwen_manifest,
        qwen3guard_manifest_path=qwen3guard_manifest,
    )

    assert report["qwen3guard"]["structured_response_validity"] == 1.0
    assert report["qwen3guard"]["strict"]["accuracy"] == 1.0


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
    qwen_manifest, qwen3guard_manifest = _write_matching_manifests(tmp_path)

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
            qwen_manifest_path=qwen_manifest,
            qwen3guard_manifest_path=qwen3guard_manifest,
        )


def test_comparison_rejects_mismatched_capture_provenance(tmp_path: Path) -> None:
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
    qwen_manifest, qwen3guard_manifest = _write_matching_manifests(tmp_path)
    payload = json.loads(qwen3guard_manifest.read_text(encoding="utf-8"))
    payload["split_sha256"]["development"] = "0" * 64
    qwen3guard_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched provenance"):
        compare_qwen_classifier_captures(
            development_cases_path=DEVELOPMENT,
            calibration_cases_path=CALIBRATION,
            qwen_predictions_path=_write_jsonl(tmp_path / "qwen.jsonl", qwen),
            qwen3guard_predictions_path=_write_jsonl(
                tmp_path / "qwen3guard.jsonl",
                qwen3guard,
            ),
            qwen_manifest_path=qwen_manifest,
            qwen3guard_manifest_path=qwen3guard_manifest,
        )
