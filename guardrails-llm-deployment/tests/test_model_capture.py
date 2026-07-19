import json
from pathlib import Path

import pytest

from guardrails_llm import model_capture
from guardrails_llm.guard_classifier import GuardClassification
from guardrails_llm.judging import JudgeResult
from guardrails_llm.model_calibration import (
    CLASSIFIER_LABELS,
    load_classifier_calibration_cases,
    load_classifier_predictions,
    load_judge_calibration_cases,
    load_judge_predictions,
)
from guardrails_llm.model_capture import run_model_calibration_capture
from guardrails_llm.model_config import (
    OpenAIModelConfig,
    RemoteModelsNotAllowedError,
)


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_CASES = ROOT / "data" / "model_classifier_calibration_v1.jsonl"
JUDGE_CASES = ROOT / "data" / "judge_calibration_v1.jsonl"
SOURCE_CASES = ROOT / "data" / "eval_cases_milestone3_holdout_v3.jsonl"
SOURCE_RESULTS = ROOT / "reports" / "disposition_guardrail_holdout_v3_results.json"


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeClassifier:
    model_name = "fake-classifier"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> GuardClassification:
        self.calls.append(text)
        if len(self.calls) == 2:
            return GuardClassification(
                label="unsafe_request",
                confidence=1.0,
                explanation="model_classifier_error:ValueError",
            )
        return GuardClassification(
            label="safe",
            confidence=0.75,
            explanation="Benign course request.",
        )


class FakeJudge:
    model_name = "fake-judge"

    def __init__(self) -> None:
        self.calls = []

    def judge(self, case, result) -> JudgeResult:
        self.calls.append((case, result))
        return JudgeResult(
            case_id=case.case_id,
            category=case.category,
            grounded=True,
            privacy_safe=True,
            injection_safe=True,
            integrity_safe=True,
            refusal_appropriate=True,
            score=1.0,
            notes=["fixture judge"],
        )


def test_stratified_classifier_selection_covers_each_label() -> None:
    cases = load_classifier_calibration_cases(CLASSIFIER_CASES)

    selected = model_capture.select_classifier_calibration_cases(
        cases,
        limit=6,
        strategy="stratified",
    )

    assert [case.expected_label for case in selected] == list(CLASSIFIER_LABELS)


def test_stratified_selection_respects_zero_limit() -> None:
    cases = load_classifier_calibration_cases(CLASSIFIER_CASES)

    selected = model_capture.select_classifier_calibration_cases(
        cases,
        limit=0,
        strategy="stratified",
    )

    assert selected == []


def test_stratified_judge_selection_covers_matched_and_mismatched_behaviors() -> None:
    cases = load_judge_calibration_cases(JUDGE_CASES)

    selected = model_capture.select_judge_calibration_cases(
        cases,
        limit=8,
        strategy="stratified",
    )

    assert [
        (
            case.expected_behavior.value,
            case.actual_behavior is case.expected_behavior,
        )
        for case in selected
    ] == [
        ("answer", True),
        ("answer", False),
        ("block", True),
        ("block", False),
        ("abstain", True),
        ("abstain", False),
        ("redirect", True),
        ("redirect", False),
    ]


def test_capture_requires_explicit_remote_approval(tmp_path: Path) -> None:
    with pytest.raises(RemoteModelsNotAllowedError, match="--allow-remote-models"):
        run_model_calibration_capture(
            component="classifier",
            config=OpenAIModelConfig(),
            classifier_output_path=tmp_path / "classifier.jsonl",
            manifest_output_path=tmp_path / "manifest.json",
        )

    assert not list(tmp_path.iterdir())


def test_classifier_capture_writes_replay_file_and_safe_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    classifier_output = tmp_path / "classifier.jsonl"
    manifest_output = tmp_path / "manifest.json"
    classifier = FakeClassifier()

    manifest = run_model_calibration_capture(
        component="classifier",
        config=_test_config(tmp_path, monkeypatch),
        classifier_cases_path=CLASSIFIER_CASES,
        source_cases_path=SOURCE_CASES,
        classifier_output_path=classifier_output,
        manifest_output_path=manifest_output,
        limit_cases=2,
        classifier=classifier,
        clock=FakeClock(1.0, 1.01, 2.0, 2.03),
        captured_at="2026-07-17T12:00:00Z",
    )

    predictions = load_classifier_predictions(classifier_output)
    stored_manifest = json.loads(manifest_output.read_text(encoding="utf-8"))

    assert len(classifier.calls) == 2
    assert len(predictions) == 2
    assert predictions[0].predicted_label == "safe"
    assert predictions[0].model == "fake-classifier"
    assert predictions[0].provider == "openai_compatible"
    assert predictions[0].latency_ms == pytest.approx(10.0)
    assert predictions[1].predicted_label is None
    assert predictions[1].error == "model_classifier_error:ValueError"
    assert manifest == stored_manifest
    assert manifest["evidence_scope"] == "live_remote_model_capture"
    assert manifest["endpoint_category"] == "custom_openai_compatible"
    assert manifest["request_policy"] == {
        "timeout_seconds": 90.0,
        "max_transport_retries": 1,
    }
    assert manifest["prompt_versions"]["classifier"] == "guard-classifier-v3.3"
    assert manifest["classifier"] == {
        "model": "fake-classifier",
        "output_path": str(classifier_output),
        "cases_requested": 2,
        "request_count": 2,
        "successful_predictions": 1,
        "failed_predictions": 1,
        "prediction_coverage": 0.5,
        "p50_latency_ms": 20.0,
        "p95_latency_ms": 30.0,
    }
    serialized = manifest_output.read_text(encoding="utf-8")
    assert "test-secret-never-store" not in serialized
    assert "internal.example" not in serialized


def test_judge_capture_reconstructs_source_case_and_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    judge_output = tmp_path / "judge.jsonl"
    manifest_output = tmp_path / "manifest.json"
    judge = FakeJudge()

    manifest = run_model_calibration_capture(
        component="judge",
        config=_test_config(tmp_path, monkeypatch),
        judge_cases_path=JUDGE_CASES,
        source_cases_path=SOURCE_CASES,
        source_results_path=SOURCE_RESULTS,
        judge_output_path=judge_output,
        manifest_output_path=manifest_output,
        limit_cases=1,
        judge=judge,
        clock=FakeClock(3.0, 3.025),
        captured_at="2026-07-17T12:00:00Z",
    )

    predictions = load_judge_predictions(judge_output)
    calibration_case_id = json.loads(
        JUDGE_CASES.read_text(encoding="utf-8").splitlines()[0]
    )["case_id"]
    source_case, source_result = judge.calls[0]

    assert len(predictions) == 1
    assert predictions[0].case_id == calibration_case_id
    assert predictions[0].model == "fake-judge"
    assert predictions[0].notes == ["fixture judge"]
    assert predictions[0].latency_ms == pytest.approx(25.0)
    assert source_case.case_id == "normal-003"
    assert source_result.case_id == "normal-003"
    assert source_result.actual_behavior.value == "answer"
    assert manifest["prompt_versions"]["judge"] == "guardrail-judge-v2.2"


def test_both_capture_validates_all_inputs_before_model_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    classifier = FakeClassifier()
    classifier_output = tmp_path / "classifier.jsonl"

    with pytest.raises(FileNotFoundError):
        run_model_calibration_capture(
            component="both",
            config=_test_config(tmp_path, monkeypatch),
            classifier_cases_path=CLASSIFIER_CASES,
            judge_cases_path=tmp_path / "missing-judge-cases.jsonl",
            source_cases_path=SOURCE_CASES,
            source_results_path=SOURCE_RESULTS,
            classifier_output_path=classifier_output,
            judge_output_path=tmp_path / "judge.jsonl",
            manifest_output_path=tmp_path / "manifest.json",
            limit_cases=1,
            classifier=classifier,
        )

    assert classifier.calls == []
    assert not classifier_output.exists()


def _test_config(tmp_path: Path, monkeypatch) -> OpenAIModelConfig:
    monkeypatch.setenv("TEST_CAPTURE_API_KEY", "test-secret-never-store")
    monkeypatch.setenv(
        "TEST_CAPTURE_BASE_URL",
        "https://internal.example/v1",
    )
    return OpenAIModelConfig(
        api_key_env="TEST_CAPTURE_API_KEY",
        base_url_env="TEST_CAPTURE_BASE_URL",
        api_url_alias_env="TEST_CAPTURE_API_URL",
        allow_remote_models=True,
        env_file=tmp_path / "missing.env",
    )
