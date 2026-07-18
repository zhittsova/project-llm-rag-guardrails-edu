import json
from pathlib import Path

import pytest

from guardrails_llm.e2e_capture import (
    evaluate_calibration_e2e_capture,
    run_calibration_e2e_capture,
)
from guardrails_llm.model_config import OpenAIModelConfig
from guardrails_llm.model_profiles import INHOUSE_EMBEDDING_MODEL, INHOUSE_LLM_MODEL
from guardrails_llm.pipeline import AssistantResponse


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data" / "eval_cases_milestone3_v2_calibration.jsonl"
HOLDOUT = ROOT / "data" / "eval_cases_milestone3_v2_holdout.jsonl"
CORPUS = ROOT / "data" / "python_course_docs.jsonl"
POLICY = ROOT / "data" / "guardrail_policy_bge_m3.toml"


class FakeAssistant:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, _question: str) -> AssistantResponse:
        self.calls += 1
        return AssistantResponse(
            answer="I do not have enough course-grounded evidence to answer that.",
            citations=[],
            cited_doc_ids=[],
            guard_triggers=["ungrounded"],
            latency_ms=1.0,
            disposition="abstain",
            grounding_supported=False,
        )


def _config() -> OpenAIModelConfig:
    return OpenAIModelConfig(
        embedding_model=INHOUSE_EMBEDDING_MODEL,
        answer_model=INHOUSE_LLM_MODEL,
        classifier_model=INHOUSE_LLM_MODEL,
        entailment_model=INHOUSE_LLM_MODEL,
        allow_remote_models=True,
    )


def _env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )


def test_calibration_capture_is_resumable_across_both_scenarios(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _env(monkeypatch)
    qwen = FakeAssistant()
    hybrid = FakeAssistant()
    output = tmp_path / "e2e.jsonl"
    manifest = tmp_path / "manifest.json"
    kwargs = {
        "config": _config(),
        "calibration_cases_path": CALIBRATION,
        "corpus_path": CORPUS,
        "policy_path": POLICY,
        "index_dir": tmp_path / "unused-index",
        "cache_path": tmp_path / "unused-cache.jsonl",
        "output_path": output,
        "manifest_path": manifest,
        "evidence_min_score": 0.42,
        "limit_cases": 3,
        "assistants": {
            "qwen_classifier_only": qwen,
            "complete_inhouse_hybrid": hybrid,
        },
    }

    first = run_calibration_e2e_capture(**kwargs)
    second = run_calibration_e2e_capture(
        **(kwargs | {
            "assistants": {
                "qwen_classifier_only": FakeAssistant(),
                "complete_inhouse_hybrid": FakeAssistant(),
            }
        })
    )

    assert qwen.calls == 3
    assert hybrid.calls == 3
    assert first["completed_runs"] == 6
    assert second["resumed_runs"] == 6
    assert len(output.read_text(encoding="utf-8").splitlines()) == 6
    serialized = manifest.read_text(encoding="utf-8")
    assert "fixture-key" not in serialized
    assert "https://" not in serialized


def test_calibration_capture_rejects_holdout_dataset(tmp_path: Path, monkeypatch) -> None:
    _env(monkeypatch)

    with pytest.raises(ValueError, match="calibration split"):
        run_calibration_e2e_capture(
            config=_config(),
            calibration_cases_path=HOLDOUT,
            corpus_path=CORPUS,
            policy_path=POLICY,
            index_dir=tmp_path / "unused-index",
            cache_path=tmp_path / "unused-cache.jsonl",
            output_path=tmp_path / "e2e.jsonl",
            manifest_path=tmp_path / "manifest.json",
            evidence_min_score=0.42,
            limit_cases=1,
            assistants={
                "qwen_classifier_only": FakeAssistant(),
                "complete_inhouse_hybrid": FakeAssistant(),
            },
        )


def test_e2e_capture_evaluation_keeps_failures_visible(tmp_path: Path, monkeypatch) -> None:
    _env(monkeypatch)
    output = tmp_path / "e2e.jsonl"
    manifest = tmp_path / "manifest.json"
    run_calibration_e2e_capture(
        config=_config(),
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        policy_path=POLICY,
        index_dir=tmp_path / "unused-index",
        cache_path=tmp_path / "unused-cache.jsonl",
        output_path=output,
        manifest_path=manifest,
        evidence_min_score=0.42,
        limit_cases=2,
        assistants={
            "qwen_classifier_only": FakeAssistant(),
            "complete_inhouse_hybrid": FakeAssistant(),
        },
    )
    rows = output.read_text(encoding="utf-8").splitlines()
    failed = json.loads(rows[0])
    failed["status"] = "error"
    failed["error"] = "capture_error:TimeoutError"
    failed["result"] = None
    rows[0] = json.dumps(failed)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = evaluate_calibration_e2e_capture(
        calibration_cases_path=CALIBRATION,
        output_path=output,
        limit_cases=2,
    )

    assert report["qwen_classifier_only"]["capture_failures"] == 1
    assert report["qwen_classifier_only"]["quality_gates"]["all_passed"] is False
    assert report["complete_inhouse_hybrid"]["capture_failures"] == 0
