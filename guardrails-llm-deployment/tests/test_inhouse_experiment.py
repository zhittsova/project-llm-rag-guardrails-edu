import json
from collections import Counter
from pathlib import Path

import pytest

from guardrails_llm.guard_classifier import GuardClassification
from guardrails_llm.inhouse_experiment import (
    build_balanced_classifier_benchmark,
    derive_classifier_label,
    run_v2_classifier_capture,
)
from guardrails_llm.model_config import OpenAIModelConfig, RemoteModelsNotAllowedError
from guardrails_llm.model_profiles import INHOUSE_LLM_MODEL


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data" / "eval_cases_milestone3_v2_development.jsonl"
CALIBRATION = ROOT / "data" / "eval_cases_milestone3_v2_calibration.jsonl"
CORPUS = ROOT / "data" / "python_course_docs.jsonl"


class EchoClassifier:
    model_name = INHOUSE_LLM_MODEL

    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> GuardClassification:
        self.calls.append(text)
        return GuardClassification(label="safe", confidence=0.9, explanation="fixture")


def _configure_inhouse(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )


def test_balanced_classifier_benchmark_has_100_cases_per_label() -> None:
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)

    labels = Counter(derive_classifier_label(case) for case in cases)
    splits = Counter(case.split for case in cases)
    assert len(cases) == 600
    assert set(labels.values()) == {100}
    assert splits == {"development": 450, "calibration": 150}
    assert len({case.case_id for case in cases}) == 600


def test_v2_capture_requires_explicit_remote_permission(tmp_path: Path, monkeypatch) -> None:
    _configure_inhouse(monkeypatch)

    with pytest.raises(RemoteModelsNotAllowedError):
        run_v2_classifier_capture(
            config=OpenAIModelConfig(classifier_model=INHOUSE_LLM_MODEL),
            development_cases_path=DEVELOPMENT,
            calibration_cases_path=CALIBRATION,
            corpus_path=CORPUS,
            output_path=tmp_path / "predictions.jsonl",
            manifest_path=tmp_path / "manifest.json",
            classifier=EchoClassifier(),
            limit_cases=1,
        )


def test_v2_capture_resumes_and_writes_safe_manifest(tmp_path: Path, monkeypatch) -> None:
    _configure_inhouse(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "manifest.json"
    classifier = EchoClassifier()
    config = OpenAIModelConfig(
        classifier_model=INHOUSE_LLM_MODEL,
        allow_remote_models=True,
    )

    first = run_v2_classifier_capture(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=output,
        manifest_path=manifest,
        classifier=classifier,
        limit_cases=3,
        captured_at="2026-07-18T12:00:00Z",
    )
    second_classifier = EchoClassifier()
    second = run_v2_classifier_capture(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=output,
        manifest_path=manifest,
        classifier=second_classifier,
        limit_cases=3,
        captured_at="2026-07-18T12:05:00Z",
    )

    assert len(classifier.calls) == 3
    assert second_classifier.calls == []
    assert first["completed_cases"] == 3
    assert second["resumed_cases"] == 3
    assert second["endpoint_host"] == "learning-services4.fokus.fraunhofer.de"
    assert second["split_case_counts"] == {"development": 3}
    serialized = manifest.read_text(encoding="utf-8")
    assert "fixture-key" not in serialized
    assert "https://" not in serialized
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_v2_capture_rejects_manifest_from_different_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "manifest.json"
    config = OpenAIModelConfig(
        classifier_model=INHOUSE_LLM_MODEL,
        allow_remote_models=True,
    )
    run_v2_classifier_capture(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=output,
        manifest_path=manifest,
        classifier=EchoClassifier(),
        limit_cases=2,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["configuration_fingerprint"] = "changed"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration does not match"):
        run_v2_classifier_capture(
            config=config,
            development_cases_path=DEVELOPMENT,
            calibration_cases_path=CALIBRATION,
            corpus_path=CORPUS,
            output_path=output,
            manifest_path=manifest,
            classifier=EchoClassifier(),
            limit_cases=2,
        )
