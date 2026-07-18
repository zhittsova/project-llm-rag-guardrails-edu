import json
from pathlib import Path
import threading
import time

import pytest

from guardrails_llm.e2e_capture import (
    _build_assistants,
    evaluate_calibration_e2e_capture,
    run_calibration_e2e_capture,
)
from guardrails_llm.guardrail_policy import GuardrailPolicy
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


class ConcurrentAssistant(FakeAssistant):
    def __init__(self, barrier: threading.Barrier, active: list[int]) -> None:
        super().__init__()
        self._barrier = barrier
        self._active = active

    def answer(self, question: str) -> AssistantResponse:
        self._active[0] += 1
        self._active[1] = max(self._active[1], self._active[0])
        self._barrier.wait(timeout=2)
        time.sleep(0.01)
        response = super().answer(question)
        self._active[0] -= 1
        return response


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
    assert first["request_policy"] == {
        "timeout_seconds": 90.0,
        "max_transport_retries": 1,
    }
    assert first["embedding_cache_mode"] == "read_only"
    assert first["expected_disposition_counts"] == {
        "abstain": 1,
        "answer": 1,
        "block": 1,
    }
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


def test_calibration_capture_uses_separate_assistants_for_bounded_concurrency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _env(monkeypatch)
    barrier = threading.Barrier(2)
    active = [0, 0]

    def build_assistants(**_kwargs):
        return {
            "qwen_classifier_only": ConcurrentAssistant(barrier, active),
            "complete_inhouse_hybrid": ConcurrentAssistant(barrier, active),
        }

    monkeypatch.setattr(
        "guardrails_llm.e2e_capture._build_assistants",
        build_assistants,
    )

    manifest = run_calibration_e2e_capture(
        config=_config(),
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        policy_path=POLICY,
        index_dir=tmp_path / "unused-index",
        cache_path=tmp_path / "unused-cache.jsonl",
        output_path=tmp_path / "e2e.jsonl",
        manifest_path=tmp_path / "manifest.json",
        evidence_min_score=0.42,
        limit_cases=2,
        max_concurrency=2,
    )

    assert active[1] == 2
    assert manifest["max_concurrency"] == 2
    assert manifest["completed_runs"] == 4


def test_build_assistants_propagates_remote_request_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = OpenAIModelConfig(
        embedding_model=INHOUSE_EMBEDDING_MODEL,
        answer_model=INHOUSE_LLM_MODEL,
        classifier_model=INHOUSE_LLM_MODEL,
        entailment_model=INHOUSE_LLM_MODEL,
        request_timeout_seconds=12.5,
        max_transport_retries=0,
        allow_remote_models=True,
    )
    embedder_calls = []
    assistant_calls = []
    fake_embedder = object()

    def fake_create_embedder(*_args, **kwargs):
        embedder_calls.append(kwargs)
        return fake_embedder

    def fake_build_assistant(*_args, **kwargs):
        assistant_calls.append(kwargs)
        return object()

    monkeypatch.setattr("guardrails_llm.e2e_capture.create_embedder", fake_create_embedder)
    monkeypatch.setattr(
        "guardrails_llm.e2e_capture.load_guardrail_policy",
        lambda *_args, **_kwargs: GuardrailPolicy.default(),
    )
    monkeypatch.setattr("guardrails_llm.e2e_capture.build_assistant", fake_build_assistant)

    _build_assistants(
        config=config,
        corpus_path=CORPUS,
        policy_path=POLICY,
        index_dir=tmp_path / "index",
        cache_path=tmp_path / "cache.jsonl",
        evidence_min_score=0.42,
        entailment_min_confidence=0.8,
        course_id="python-intro",
    )

    assert embedder_calls[0]["model_config"] is config
    assert len(assistant_calls) == 2
    assert all(call["model_config"] is config for call in assistant_calls)


def test_e2e_capture_checkpoints_workers_in_completion_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _env(monkeypatch)
    selected_ids = []

    def fake_build_assistants(**_kwargs):
        return {
            "qwen_classifier_only": object(),
            "complete_inhouse_hybrid": object(),
        }

    def fake_capture_one_run(work):
        scenario, case, _assistant = work
        selected_ids.append(case.case_id)
        time.sleep(0.05 if len(selected_ids) == 1 else 0.01)
        return {
            "schema_version": 1,
            "scenario": scenario,
            "case_id": case.case_id,
            "status": "error",
            "error": "fixture",
            "result": None,
        }

    monkeypatch.setattr(
        "guardrails_llm.e2e_capture._build_assistants",
        fake_build_assistants,
    )
    monkeypatch.setattr(
        "guardrails_llm.e2e_capture._capture_one_run",
        fake_capture_one_run,
    )
    output = tmp_path / "e2e.jsonl"

    run_calibration_e2e_capture(
        config=_config(),
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        policy_path=POLICY,
        index_dir=tmp_path / "unused-index",
        cache_path=tmp_path / "unused-cache.jsonl",
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        evidence_min_score=0.42,
        limit_cases=2,
        max_concurrency=2,
    )

    first_row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first_row["case_id"] == selected_ids[1]


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
        manifest_path=manifest,
        limit_cases=2,
    )

    assert report["qwen_classifier_only"]["capture_failures"] == 1
    assert report["qwen_classifier_only"]["quality_gates"]["all_passed"] is False
    assert report["complete_inhouse_hybrid"]["capture_failures"] == 0
    intervals = report["complete_inhouse_hybrid"]["confidence_intervals"]
    assert intervals["row"]["sampling_units"] == 2
    assert intervals["family"]["sampling_units"] <= 2


def test_e2e_evaluation_rejects_manifest_with_invalid_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["configuration_fingerprint"] = "forged"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration fingerprint"):
        evaluate_calibration_e2e_capture(
            calibration_cases_path=CALIBRATION,
            output_path=output,
            manifest_path=manifest,
            limit_cases=2,
        )


def test_e2e_evaluation_rejects_rows_outside_manifest_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    stale = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    stale["case_id"] = "calibration-stale-case"
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stale) + "\n")

    with pytest.raises(ValueError, match="outside the selected cases"):
        evaluate_calibration_e2e_capture(
            calibration_cases_path=CALIBRATION,
            output_path=output,
            manifest_path=manifest,
            limit_cases=2,
        )


def test_e2e_evaluation_rebuilds_expected_fields_from_selected_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    baseline = evaluate_calibration_e2e_capture(
        calibration_cases_path=CALIBRATION,
        output_path=output,
        manifest_path=manifest,
        limit_cases=2,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    successful = next(row for row in rows if row["status"] == "success")
    successful["result"].update(
        {
            "case_id": "forged-case-id",
            "category": "forged-category",
            "should_answer": True,
            "passed": True,
            "expected_behavior": "answer",
            "attack_type": "forged_attack",
            "difficulty": "hard",
            "split": "holdout",
            "family_id": "forged-family",
            "language": "de",
            "expected_doc_ids": ["forged-document"],
            "evidence_available": True,
            "required_claims": ["forged claim"],
        }
    )
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    replayed = evaluate_calibration_e2e_capture(
        calibration_cases_path=CALIBRATION,
        output_path=output,
        manifest_path=manifest,
        limit_cases=2,
    )

    assert replayed == baseline
