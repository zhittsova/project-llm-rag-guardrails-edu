from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

import guardrails_llm.evidence_refresh as evidence_refresh
from guardrails_llm.evidence_refresh import refresh_calibration_evidence
from guardrails_llm.final_evidence import FinalEvidenceError


DETERMINISTIC_SCENARIOS = (
    "baseline",
    "normalized_regex_guardrails",
    "fuzzy_plus_shared_controls",
    "similarity_plus_shared_controls",
    "hybrid_policy_guardrails",
)


def test_refresh_calibration_evidence_writes_all_reports(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)

    report = refresh_calibration_evidence(
        **inputs,
        deterministic_output=tmp_path / "deterministic.json",
        model_output=tmp_path / "model.json",
        failure_output=tmp_path / "failures.json",
        final_json_output=tmp_path / "final.json",
        final_markdown_output=tmp_path / "final.md",
    )

    assert report["cases"] == 400
    assert set(report["techniques"]) == {
        "baseline",
        "regex_only_with_shared_controls",
        "fuzzy_only_with_shared_controls",
        "bge_similarity_with_shared_controls",
        "deterministic_hybrid",
        "qwen_classifier_only",
        "complete_inhouse_hybrid",
    }
    assert report["techniques"]["complete_inhouse_hybrid"]["correct"] == 392
    assert report["primary_quality_gates_passed"] is True
    assert report["expected_document_diagnostic_passed"] is False

    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert model["capture"]["completed_runs"] == 800
    assert model["quality_gate_interpretation"] == {
        "agreed_primary_gates_passed": True,
        "extra_expected_document_citation_gate_passed": False,
    }

    failures = json.loads(
        (tmp_path / "failures.json").read_text(encoding="utf-8")
    )
    assert failures["failed_cases"] == 8
    assert failures["stage_counts"] == {
        "answerability_rejected_with_expected_document_present": 2,
        "entailment_rejected_unsupported_extra_claims": 4,
        "expected_policy_document_not_retrieved": 2,
        "other": 0,
    }
    assert failures["attack_types"] == {"normal_course_positive_direct": 8}
    assert "coverage_role" not in failures


def test_refresh_rejects_incomplete_model_capture(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    manifest = json.loads(inputs["model_manifest_path"].read_text(encoding="utf-8"))
    manifest["completed_runs"] = 799
    inputs["model_manifest_path"].write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(FinalEvidenceError, match="800 successful runs"):
        refresh_calibration_evidence(
            **inputs,
            deterministic_output=tmp_path / "deterministic.json",
            model_output=tmp_path / "model.json",
            failure_output=tmp_path / "failures.json",
            final_json_output=tmp_path / "final.json",
            final_markdown_output=tmp_path / "final.md",
        )


def test_refresh_rejects_duplicate_model_capture_rows(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    rows = inputs["model_capture_path"].read_text(encoding="utf-8").splitlines()
    rows[-1] = rows[-2]
    inputs["model_capture_path"].write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    with pytest.raises(FinalEvidenceError, match="duplicate run keys"):
        refresh_calibration_evidence(
            **inputs,
            deterministic_output=tmp_path / "deterministic.json",
            model_output=tmp_path / "model.json",
            failure_output=tmp_path / "failures.json",
            final_json_output=tmp_path / "final.json",
            final_markdown_output=tmp_path / "final.md",
        )


def test_refresh_rejects_capture_outside_manifest_selection(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    rows = [
        json.loads(line)
        for line in inputs["model_capture_path"].read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["case_id"] = "holdout-case"
    rows[-1]["result"]["case_id"] = "holdout-case"
    inputs["model_capture_path"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(FinalEvidenceError, match="selected calibration run keys"):
        refresh_calibration_evidence(
            **inputs,
            deterministic_output=tmp_path / "deterministic.json",
            model_output=tmp_path / "model.json",
            failure_output=tmp_path / "failures.json",
            final_json_output=tmp_path / "final.json",
            final_markdown_output=tmp_path / "final.md",
        )


def test_refresh_rejects_modified_deterministic_capture(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    inputs["deterministic_path"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(FinalEvidenceError, match="deterministic capture hash"):
        refresh_calibration_evidence(
            **inputs,
            deterministic_output=tmp_path / "deterministic.json",
            model_output=tmp_path / "model.json",
            failure_output=tmp_path / "failures.json",
            final_json_output=tmp_path / "final.json",
            final_markdown_output=tmp_path / "final.md",
        )


def test_refresh_rebuilds_stale_model_evaluation(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    inputs["model_evaluation_path"].write_text(
        json.dumps({"stale": True}), encoding="utf-8"
    )

    refresh_calibration_evidence(
        **inputs,
        deterministic_output=tmp_path / "deterministic.json",
        model_output=tmp_path / "model.json",
        failure_output=tmp_path / "failures.json",
        final_json_output=tmp_path / "final.json",
        final_markdown_output=tmp_path / "final.md",
    )

    rebuilt = json.loads(
        inputs["model_evaluation_path"].read_text(encoding="utf-8")
    )
    assert rebuilt["complete_inhouse_hybrid"]["successful_cases"] == 400
    assert rebuilt["_provenance"]["holdout_used"] is False
    assert rebuilt["_provenance"]["source_capture_sha256"] == sha256(
        inputs["model_capture_path"].read_bytes()
    ).hexdigest()


def test_refresh_does_not_publish_when_consolidation_fails(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _write_inputs(tmp_path)
    outputs = {
        "deterministic_output": tmp_path / "deterministic.json",
        "model_output": tmp_path / "model.json",
        "failure_output": tmp_path / "failures.json",
        "final_json_output": tmp_path / "final.json",
        "final_markdown_output": tmp_path / "final.md",
    }
    for path in outputs.values():
        path.write_text("previous\n", encoding="utf-8")
    inputs["model_evaluation_path"].write_text("previous\n", encoding="utf-8")

    def fail_consolidation(**_kwargs):
        raise FinalEvidenceError("late validation failure")

    monkeypatch.setattr(
        evidence_refresh, "write_calibration_evidence", fail_consolidation
    )

    with pytest.raises(FinalEvidenceError, match="late validation failure"):
        refresh_calibration_evidence(**inputs, **outputs)

    for path in (*outputs.values(), inputs["model_evaluation_path"]):
        assert path.read_text(encoding="utf-8") == "previous\n"


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    deterministic_path = tmp_path / "deterministic-raw.json"
    deterministic_path.write_text(
        json.dumps(
            {
                scenario: _summary(accuracy=0.75)
                for scenario in DETERMINISTIC_SCENARIOS
            }
        ),
        encoding="utf-8",
    )

    deterministic_manifest_path = tmp_path / "deterministic-manifest.json"
    deterministic_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_scope": "calibration_only",
                "dataset_version": "milestone3-v2",
                "dataset_manifest_sha256": "a" * 64,
                "calibration_split_sha256": "b" * 64,
                "source_report_sha256": sha256(
                    deterministic_path.read_bytes()
                ).hexdigest(),
                "cases": 400,
                "holdout_used": False,
            }
        ),
        encoding="utf-8",
    )

    model_evaluation_path = tmp_path / "model-evaluation.json"
    model_evaluation_path.write_text(
        json.dumps(
            {
                "qwen_classifier_only": _scenario(accuracy=0.97),
                "complete_inhouse_hybrid": _scenario(accuracy=0.98),
            }
        ),
        encoding="utf-8",
    )

    model_manifest_path = tmp_path / "model-manifest.json"
    model_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "complete",
                "dataset_version": "milestone3-v2",
                "dataset_manifest_sha256": "a" * 64,
                "calibration_split_sha256": "b" * 64,
                "runtime_config_sha256": "c" * 64,
                "configuration_fingerprint": "d" * 64,
                "expected_runs": 800,
                "selected_case_ids": [
                    f"case-{index:03d}" for index in range(400)
                ],
                "selected_run_keys": [
                    [scenario, f"case-{index:03d}"]
                    for scenario in (
                        "qwen_classifier_only",
                        "complete_inhouse_hybrid",
                    )
                    for index in range(400)
                ],
                "completed_runs": 800,
                "failed_runs": 0,
                "expected_disposition_counts": {
                    "answer": 100,
                    "block": 100,
                    "abstain": 100,
                    "redirect": 100,
                },
                "models": {
                    "embedding": "BAAI/bge-m3",
                    "answer": "Qwen",
                    "classifier": "Qwen",
                    "entailment": "Qwen",
                },
                "prompt_versions": {
                    "answer": "answer-v1",
                    "classifier": "classifier-v1",
                    "entailment": "entailment-v1",
                },
                "retrieval": {
                    "top_k": 8,
                    "policy_context_top_k": 2,
                    "policy_context_min_score": 0.51,
                },
                "thresholds": {
                    "retrieval_evidence": 0.52,
                    "classifier_confidence": 0.65,
                    "entailment_confidence": 0.8,
                },
                "holdout_used": False,
            }
        ),
        encoding="utf-8",
    )

    model_capture_path = tmp_path / "model-capture.jsonl"
    rows = []
    for scenario in ("qwen_classifier_only", "complete_inhouse_hybrid"):
        for index in range(400):
            disposition = ("answer", "block", "abstain", "redirect")[index // 100]
            should_answer = disposition in {"answer", "redirect"}
            result = {
                "case_id": f"case-{index:03d}",
                "passed": True,
                "category": "normal_course" if disposition == "answer" else disposition,
                "should_answer": should_answer,
                "answered": should_answer,
                "triggers": [],
                "citations": (
                    ["course-policy", "alternate-policy"]
                    if disposition == "answer"
                    else []
                ),
                "latency_ms": 1.0,
                "answer": "Grounded answer." if disposition == "answer" else "Policy response.",
                "expected_behavior": disposition,
                "actual_behavior": disposition,
                "expected_doc_ids": ["course-policy"] if disposition == "answer" else [],
                "evidence_available": disposition == "answer",
                "retrieved_doc_ids": ["course-policy"] if disposition == "answer" else [],
                "retrieved_chunks": ["course-policy:lc:1"] if disposition == "answer" else [],
                "retrieval_scores": {"course-policy:lc:1": 0.9} if disposition == "answer" else {},
                "retrieved_evidence": (
                    [
                        {
                            "chunk_id": "course-policy:lc:1",
                            "doc_id": "course-policy",
                            "text": "Grounded answer.",
                        },
                        {
                            "chunk_id": "alternate-policy:lc:1",
                            "doc_id": "alternate-policy",
                            "text": "Additional grounded evidence.",
                        },
                    ]
                    if disposition == "answer"
                    else []
                ),
                "cited_doc_ids": (
                    ["course-policy", "alternate-policy"]
                    if disposition == "answer"
                    else []
                ),
                "supporting_chunks": (
                    ["course-policy:lc:1", "alternate-policy:lc:1"]
                    if disposition == "answer"
                    else []
                ),
                "grounding_supported": True if disposition == "answer" else None,
                "grounding_confidence": 0.9 if disposition == "answer" else None,
                "grounding_error": None,
                "unsupported_claims": [],
                "attack_type": "normal_course_positive_direct",
                "difficulty": "medium",
                "split": "calibration",
                "family_id": f"family-{index // 25:02d}",
                "language": "en",
                "required_claims": [],
            }
            failure_count = 11 if scenario == "qwen_classifier_only" else 8
            if disposition == "answer" and index < failure_count:
                result.update(
                    passed=False,
                    actual_behavior="abstain",
                    answered=False,
                    retrieved_doc_ids=[],
                    retrieved_chunks=[],
                    cited_doc_ids=[],
                    citations=[],
                    supporting_chunks=[],
                    grounding_supported=False,
                )
                if scenario == "complete_inhouse_hybrid" and 2 <= index < 4:
                    result.update(
                        retrieved_doc_ids=["course-policy"],
                        retrieved_chunks=["course-policy:lc:1"],
                    )
                if scenario == "complete_inhouse_hybrid" and 4 <= index < 8:
                    result.update(
                        retrieved_doc_ids=["course-policy"],
                        retrieved_chunks=["course-policy:lc:1"],
                        unsupported_claims=["unsupported extra claim"],
                    )
            rows.append(
                {
                    "schema_version": 1,
                    "scenario": scenario,
                    "case_id": result["case_id"],
                    "status": "success",
                    "error": None,
                    "result": result,
                }
            )
    model_capture_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "deterministic_path": deterministic_path,
        "deterministic_manifest_path": deterministic_manifest_path,
        "model_evaluation_path": model_evaluation_path,
        "model_manifest_path": model_manifest_path,
        "model_capture_path": model_capture_path,
    }


def _scenario(*, accuracy: float) -> dict[str, object]:
    return {
        "expected_cases": 400,
        "successful_cases": 400,
        "capture_failures": 0,
        "missing_cases": 0,
        "metrics_on_successful_cases": _summary(accuracy=accuracy),
        "confidence_intervals": _confidence_intervals(accuracy),
        "quality_gates": {
            "complete_capture": True,
            "behavior_accuracy_passed": True,
            "macro_f1_passed": True,
            "answer_recall_passed": True,
            "block_recall_passed": True,
            "abstain_recall_passed": True,
            "redirect_recall_passed": True,
            "safe_false_refusal_passed": True,
            "supported_answer_precision_passed": True,
            "citation_entailment_precision_passed": True,
            "expected_document_citation_precision_passed": False,
            "all_passed": False,
        },
    }


def _summary(*, accuracy: float) -> dict[str, object]:
    return {
        "total": 400,
        "passed": round(400 * accuracy),
        "behavior_accuracy": accuracy,
        "macro_behavior_f1": accuracy,
        "behavior_metrics": {
            disposition: {
                "support": 100,
                "predicted": 100,
                "true_positives": 100,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
            for disposition in ("answer", "block", "abstain", "redirect")
        },
        "safe_false_refusal_rate": 0.04,
        "false_unsafe_answer_rate": 0.0,
        "retrieval_recall_at_3": 0.925,
        "supported_answer_precision": 1.0,
        "expected_document_citation_precision": 0.766,
        "citation_entailment_precision": 1.0,
        "claim_support_rate": 1.0,
        "confidence_intervals": _confidence_intervals(accuracy),
    }


def _confidence_intervals(accuracy: float) -> dict[str, object]:
    metric = {"point": accuracy, "lower": accuracy - 0.02, "upper": 1.0}
    return {
        "row": {"metrics": {"behavior_accuracy": metric}},
        "family": {"metrics": {"behavior_accuracy": metric}},
    }
