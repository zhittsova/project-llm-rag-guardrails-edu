from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails_llm.workshop3_demo import write_workshop3_demo


def test_offline_demo_renders_guardrail_pipeline_and_calibration_metrics(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence()), encoding="utf-8")
    output = tmp_path / "demo.html"

    result = write_workshop3_demo(
        evidence_path=evidence,
        output_path=output,
    )

    html = output.read_text(encoding="utf-8")
    assert result["mode"] == "offline"
    assert result["scenarios"] == 5
    assert "Workshop 3 Guardrail Demo" in html
    assert "Calibration evidence only" in html
    assert "BAAI/bge-m3" in html
    assert "Qwen/Qwen3.6-35B-A3B" in html
    assert "Prompt injection" in html
    assert "Academic integrity" in html
    assert "97.8%" in html
    assert "25.0%" in html
    assert "9 false abstentions" in html
    assert "The frozen holdout remains unopened" in html
    assert all(line == line.rstrip() for line in html.splitlines())


def test_demo_rejects_holdout_derived_evidence(tmp_path: Path) -> None:
    payload = _evidence()
    payload["holdout_used"] = True
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration-only"):
        write_workshop3_demo(
            evidence_path=evidence,
            output_path=tmp_path / "demo.html",
        )


def test_live_demo_requires_explicit_remote_permission(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence()), encoding="utf-8")

    with pytest.raises(ValueError, match="allow-remote-models"):
        write_workshop3_demo(
            evidence_path=evidence,
            output_path=tmp_path / "demo.html",
            live=True,
            allow_remote_models=False,
        )


def test_live_demo_renders_captured_baseline_and_hybrid_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence()), encoding="utf-8")
    output = tmp_path / "demo.html"

    monkeypatch.setattr(
        "guardrails_llm.workshop3_demo._run_live_scenarios",
        lambda **_kwargs: [
            {
                "scenario_id": "safe_course_answer",
                "baseline": {
                    "disposition": "answer",
                    "answer": "Baseline answer",
                    "triggers": [],
                    "citations": ["Lecture (lec04)"],
                    "grounding_supported": None,
                },
                "hybrid": {
                    "disposition": "answer",
                    "answer": "Grounded hybrid answer",
                    "triggers": [],
                    "citations": ["Lecture (lec04)"],
                    "grounding_supported": True,
                },
            }
        ],
    )

    result = write_workshop3_demo(
        evidence_path=evidence,
        output_path=output,
        live=True,
        allow_remote_models=True,
    )

    html = output.read_text(encoding="utf-8")
    assert result["mode"] == "live"
    assert "Baseline answer" in html
    assert "Grounded hybrid answer" in html
    assert "Live Fraunhofer run" in html


def _evidence() -> dict[str, object]:
    techniques = {}
    for technique, accuracy, macro_f1 in (
        ("baseline", 0.25, 0.1),
        ("regex_only_with_shared_controls", 0.5075, 0.4204),
        ("fuzzy_only_with_shared_controls", 0.5375, 0.4849),
        ("bge_similarity_with_shared_controls", 0.79, 0.7878),
        ("deterministic_hybrid", 0.7925, 0.7899),
        ("qwen_classifier_only", 0.985, 0.985),
        ("complete_inhouse_hybrid", 0.978, 0.978),
    ):
        techniques[technique] = {
            "behavior_accuracy": accuracy,
            "macro_f1": macro_f1,
            "safe_false_refusal_rate": 0.045,
            "false_unsafe_answer_rate": 0.0 if "qwen" in technique or "complete" in technique else 0.2,
        }
    return {
        "schema_version": 1,
        "evidence_scope": "calibration_only",
        "cases": 400,
        "holdout_used": False,
        "primary_quality_gates_passed": True,
        "models": {
            "embedding": "BAAI/bge-m3",
            "answer": "Qwen/Qwen3.6-35B-A3B",
            "classifier": "Qwen/Qwen3.6-35B-A3B",
            "entailment": "Qwen/Qwen3.6-35B-A3B",
        },
        "techniques": techniques,
        "failure_analysis": {
            "failed_cases": 9,
            "failure_disposition": "false_abstention",
            "stage_counts": {
                "expected_policy_document_not_retrieved": 2,
                "answerability_rejected_with_expected_document_present": 3,
                "entailment_rejected_unsupported_extra_claims": 4,
            },
        },
        "limitations": [
            "This is calibration evidence; the frozen holdout remains unopened."
        ],
    }
