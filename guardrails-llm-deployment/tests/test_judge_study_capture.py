from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails_llm.judge_study_capture import run_judge_study_capture
from guardrails_llm.judging import JudgeResult
from guardrails_llm.model_calibration import load_judge_predictions
from guardrails_llm.model_config import OpenAIModelConfig, RemoteModelsNotAllowedError


class FakeJudge:
    calls = 0

    def __init__(self, config: OpenAIModelConfig) -> None:
        self.model_name = config.judge_model

    def judge(self, case, result) -> JudgeResult:
        type(self).calls += 1
        return JudgeResult(
            case_id=case.case_id,
            category=case.category,
            grounded=True,
            privacy_safe=True,
            injection_safe=True,
            integrity_safe=True,
            refusal_appropriate=True,
            score=1.0,
            notes=["fake judge"],
        )


def test_capture_requires_explicit_remote_model_approval(tmp_path: Path) -> None:
    with pytest.raises(RemoteModelsNotAllowedError, match="--allow-remote-models"):
        run_judge_study_capture(
            config=OpenAIModelConfig(),
            study_dir=tmp_path,
            source_cases_path=tmp_path / "cases.jsonl",
            source_results_path=tmp_path / "results.json",
            output_path=tmp_path / "predictions.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )

    assert list(tmp_path.iterdir()) == []


def test_capture_is_label_blind_resumable_and_credential_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-never-store")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    monkeypatch.setattr(
        "guardrails_llm.judge_study_capture.OpenAIJudge",
        FakeJudge,
    )
    FakeJudge.calls = 0
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "judge_study_manifest.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    _write_jsonl(
        study_dir / "judge_study_mapping.jsonl",
        [
            {
                "item_id": "judge_calibration-one",
                "judge_split": "judge_calibration",
                "source_case_id": "case-one",
                "parent_case_id": "family-one",
                "scenario": "scenario-one",
            },
            {
                "item_id": "judge_validation-two",
                "judge_split": "judge_validation",
                "source_case_id": "case-two",
                "parent_case_id": "family-two",
                "scenario": "scenario-one",
            },
        ],
    )
    source_cases = tmp_path / "cases.jsonl"
    _write_jsonl(
        source_cases,
        [
            _case("case-one", "answer"),
            _case("case-two", "block"),
        ],
    )
    source_results = tmp_path / "results.json"
    source_results.write_text(
        json.dumps(
            {
                "scenario-one": [
                    _result("case-one", "answer"),
                    _result("case-two", "block"),
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    manifest_path = tmp_path / "manifest.json"
    config = OpenAIModelConfig(
        judge_model="fake-model",
        allow_remote_models=True,
    )

    manifest = run_judge_study_capture(
        config=config,
        study_dir=study_dir,
        source_cases_path=source_cases,
        source_results_path=source_results,
        output_path=output,
        manifest_path=manifest_path,
        max_concurrency=2,
    )
    repeated = run_judge_study_capture(
        config=config,
        study_dir=study_dir,
        source_cases_path=source_cases,
        source_results_path=source_results,
        output_path=output,
        manifest_path=manifest_path,
        max_concurrency=2,
    )

    predictions = load_judge_predictions(output)
    serialized = manifest_path.read_text(encoding="utf-8")
    assert FakeJudge.calls == 2
    assert len(predictions) == 2
    assert manifest == repeated
    assert manifest["complete"] is True
    assert manifest["structured_response_validity"] == 1.0
    assert manifest["human_labels_used"] is False
    assert manifest["holdout_used"] is False
    assert "secret-never-store" not in serialized


def _case(case_id: str, expected_behavior: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "category": "normal_course",
        "question": "Example question",
        "expected_behavior": expected_behavior,
        "difficulty": "easy",
        "split": "calibration",
        "family_id": "family",
        "coverage_role": "positive_direct",
        "language": "en",
        "parent_case_id": "parent",
        "provenance": "test",
        "expected_doc_ids": [],
        "evidence_available": expected_behavior == "answer",
        "required_claims": [],
        "annotation_status": "generated",
    }


def _result(case_id: str, behavior: str) -> dict[str, object]:
    answered = behavior in {"answer", "redirect"}
    return {
        "case_id": case_id,
        "category": "normal_course",
        "should_answer": answered,
        "answered": answered,
        "passed": True,
        "triggers": [],
        "citations": ["Example"] if answered else [],
        "latency_ms": 1.0,
        "answer": "Example answer",
        "expected_behavior": behavior,
        "actual_behavior": behavior,
    }


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
