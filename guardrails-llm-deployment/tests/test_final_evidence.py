from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from guardrails_llm.cli import main
from guardrails_llm.final_evidence import (
    FinalEvidenceError,
    assess_final_readiness,
    build_calibration_evidence,
    render_calibration_report,
    seal_runtime_configuration,
    write_calibration_evidence,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _deterministic_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_version": "milestone3-v2",
        "cases": 400,
        "cases_per_disposition": 100,
        "holdout_used": False,
        "evidence_scope": "calibration_only",
        "techniques": {
            "baseline": {
                "behavior_accuracy": 0.25,
                "macro_f1": 0.10,
                "false_unsafe_answer_rate": 1.0,
            },
            "regex_only_with_shared_controls": {
                "behavior_accuracy": 0.5075,
                "macro_f1": 0.4204,
            },
            "fuzzy_only_with_shared_controls": {
                "behavior_accuracy": 0.5375,
                "macro_f1": 0.4849,
            },
            "bge_similarity_with_shared_controls": {
                "behavior_accuracy": 0.79,
                "macro_f1": 0.7878,
            },
            "deterministic_hybrid": {
                "behavior_accuracy": 0.7925,
                "macro_f1": 0.7899,
            },
            "qwen_classifier_only": {
                "behavior_accuracy": 0.97,
                "macro_f1": 0.9699,
            },
            "complete_inhouse_hybrid": {
                "behavior_accuracy": 0.97,
                "macro_f1": 0.9699,
            },
        },
    }


def _model_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_version": "milestone3-v2",
        "cases_per_scenario": 400,
        "cases_per_disposition": 100,
        "holdout_used": False,
        "evidence_scope": "calibration_only",
        "scenarios": {
            "qwen_classifier_only": {
                "correct": 394,
                "behavior_accuracy": 0.985,
                "macro_f1": 0.985,
                "answer_recall": 0.94,
                "block_recall": 1.0,
                "abstain_recall": 1.0,
                "redirect_recall": 1.0,
                "safe_false_refusal_rate": 0.03,
                "false_unsafe_answer_rate": 0.0,
                "retrieval_recall_at_3": 0.905,
                "supported_answer_precision": 1.0,
                "citation_entailment_precision": 1.0,
                "expected_document_citation_precision": 0.758,
                "row_accuracy_95ci": [0.9725, 0.995],
                "family_accuracy_95ci": [0.9787, 0.9917],
            },
            "complete_inhouse_hybrid": {
                "correct": 391,
                "behavior_accuracy": 0.978,
                "macro_f1": 0.978,
                "answer_recall": 0.91,
                "block_recall": 1.0,
                "abstain_recall": 1.0,
                "redirect_recall": 1.0,
                "safe_false_refusal_rate": 0.045,
                "false_unsafe_answer_rate": 0.0,
                "retrieval_recall_at_3": 0.905,
                "supported_answer_precision": 1.0,
                "citation_entailment_precision": 1.0,
                "expected_document_citation_precision": 0.752,
                "row_accuracy_95ci": [0.9625, 0.9925],
                "family_accuracy_95ci": [0.9631, 0.9942],
            },
        },
        "capture": {
            "expected_runs": 800,
            "completed_runs": 800,
            "failed_runs": 0,
            "configuration_fingerprint": "1" * 64,
        },
        "configuration": {
            "answer_prompt": "rag-answer-v2.4",
            "classifier_prompt": "guard-classifier-v3.4",
            "entailment_prompt": "answer-entailment-v1.4",
        },
        "models": {
            "embedding": "BAAI/bge-m3",
            "answer": "Qwen/Qwen3.6-35B-A3B",
        },
        "quality_gate_interpretation": {
            "agreed_primary_gates_passed": True,
            "extra_expected_document_citation_gate_passed": False,
        },
        "limitations": ["Calibration evidence only."],
    }


def _failure_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evidence_scope": "complete_inhouse_hybrid_calibration",
        "dataset_version": "milestone3-v2",
        "cases": 400,
        "failed_cases": 9,
        "failure_disposition": "false_abstention",
        "stage_counts": {
            "expected_policy_document_not_retrieved": 2,
            "answerability_rejected_with_expected_document_present": 3,
            "entailment_rejected_unsupported_extra_claims": 4,
        },
        "holdout_used": False,
    }


def test_build_calibration_evidence_uses_latest_model_scenarios(tmp_path: Path) -> None:
    report = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )

    assert report["cases"] == 400
    assert report["holdout_used"] is False
    assert report["techniques"]["baseline"]["behavior_accuracy"] == 0.25
    assert report["techniques"]["qwen_classifier_only"]["correct"] == 394
    assert report["techniques"]["complete_inhouse_hybrid"]["correct"] == 391
    assert report["primary_quality_gates_passed"] is True
    assert report["expected_document_diagnostic_passed"] is False


def test_build_calibration_evidence_accepts_versioned_common_split_scope(
    tmp_path: Path,
) -> None:
    model = _model_report()
    model["evidence_scope"] = "inhouse_common_split_calibration"

    report = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", model),
    )

    assert report["evidence_scope"] == "calibration_only"


def test_build_calibration_evidence_rejects_holdout_or_mismatched_dataset(
    tmp_path: Path,
) -> None:
    deterministic = _deterministic_report()
    deterministic["holdout_used"] = True
    with pytest.raises(FinalEvidenceError, match="holdout"):
        build_calibration_evidence(
            deterministic_path=_write_json(tmp_path / "deterministic.json", deterministic),
            model_path=_write_json(tmp_path / "model.json", _model_report()),
        )

    deterministic["holdout_used"] = False
    model = _model_report()
    model["dataset_version"] = "different"
    with pytest.raises(FinalEvidenceError, match="dataset version"):
        build_calibration_evidence(
            deterministic_path=_write_json(tmp_path / "deterministic.json", deterministic),
            model_path=_write_json(tmp_path / "model.json", model),
        )


def test_build_calibration_evidence_rejects_invalid_capture_fingerprint(
    tmp_path: Path,
) -> None:
    model = _model_report()
    model["capture"]["configuration_fingerprint"] = "not-a-sha256"

    with pytest.raises(FinalEvidenceError, match="fingerprint"):
        build_calibration_evidence(
            deterministic_path=_write_json(
                tmp_path / "deterministic.json", _deterministic_report()
            ),
            model_path=_write_json(tmp_path / "model.json", model),
        )


def test_render_calibration_report_keeps_failed_diagnostic_visible(tmp_path: Path) -> None:
    report = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
        failure_path=_write_json(tmp_path / "failures.json", _failure_report()),
    )

    markdown = render_calibration_report(report)

    assert "391/400" in markdown
    assert "0.978" in markdown
    assert "Expected-document citation precision" in markdown
    assert "0.752" in markdown
    assert "Calibration evidence" in markdown
    assert "frozen holdout remains unopened" in markdown
    assert "2 retrieval misses" in markdown
    assert "3 evidence-gate rejects" in markdown
    assert "4 entailment rejects" in markdown
    assert "Row-level accuracy 95% CI" in markdown
    assert "Family-level accuracy 95% CI" in markdown


def test_build_calibration_evidence_rejects_holdout_failure_analysis(
    tmp_path: Path,
) -> None:
    failures = _failure_report()
    failures["holdout_used"] = True

    with pytest.raises(FinalEvidenceError, match="failure analysis.*holdout"):
        build_calibration_evidence(
            deterministic_path=_write_json(
                tmp_path / "deterministic.json", _deterministic_report()
            ),
            model_path=_write_json(tmp_path / "model.json", _model_report()),
            failure_path=_write_json(tmp_path / "failures.json", failures),
        )


def _dataset_manifest(*, ready: bool) -> dict[str, object]:
    return {
        "dataset_version": "milestone3-v2",
        "holdout_frozen": True,
        "annotation_sealed": ready,
        "holdout_review_status": "adjudicated" if ready else "pending_double_review",
        "holdout_reviewed_cases": 400 if ready else 0,
        "annotation_summary": {
            "double_labeled_cases": 400 if ready else 0,
            "adjudicated_cases": 400 if ready else 0,
            "ready_for_final_holdout": ready,
        },
    }


def _judge_report(*, passing: bool) -> dict[str, object]:
    return {
        "human_labels_are_ground_truth": True,
        "models": {
            "judge-model": {
                "splits": {
                    "judge_validation": {
                        "summary": {
                            "total": 200,
                            "evaluated_predictions": 200,
                            "parse_failures": 0,
                            "missing_predictions": 0,
                        },
                        "quality_gates": {"all_passed": passing},
                    }
                }
            }
        },
    }


def _configuration_manifest(*, sealed: bool) -> dict[str, object]:
    return {
        "sealed": sealed,
        "holdout_used": False,
        "dataset_version": "milestone3-v2",
        "configuration_fingerprint": "e" * 64 if sealed else "",
        "artifacts": {
            "dataset_manifest": {"sha256": "f" * 64},
            "policy": {"sha256": "a" * 64},
            "corpus": {"sha256": "b" * 64},
            "index_manifest": {"sha256": "c" * 64},
            "calibration_evidence": {"sha256": "d" * 64},
        },
    }


def test_readiness_fails_closed_with_actionable_missing_checks(tmp_path: Path) -> None:
    calibration = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )
    report = assess_final_readiness(
        dataset_manifest=_dataset_manifest(ready=False),
        judge_report=_judge_report(passing=False),
        selected_judge_model="judge-model",
        calibration_report=calibration,
        configuration_manifest=_configuration_manifest(sealed=False),
    )

    assert report["ready"] is False
    assert "holdout_annotations_sealed" in report["failed_checks"]
    assert "judge_validation_gates_passed" in report["failed_checks"]
    assert "runtime_configuration_sealed" in report["failed_checks"]


def test_readiness_passes_only_after_all_human_and_configuration_gates(
    tmp_path: Path,
) -> None:
    calibration = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )
    report = assess_final_readiness(
        dataset_manifest=_dataset_manifest(ready=True),
        judge_report=_judge_report(passing=True),
        selected_judge_model="judge-model",
        calibration_report=calibration,
        configuration_manifest=_configuration_manifest(sealed=True),
    )

    assert report["ready"] is True
    assert report["failed_checks"] == []
    assert all(report["checks"].values())


def test_readiness_rejects_non_hex_artifact_digest(tmp_path: Path) -> None:
    calibration = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )
    configuration = _configuration_manifest(sealed=True)
    configuration["artifacts"]["policy"]["sha256"] = "g" * 64

    report = assess_final_readiness(
        dataset_manifest=_dataset_manifest(ready=True),
        judge_report=_judge_report(passing=True),
        selected_judge_model="judge-model",
        calibration_report=calibration,
        configuration_manifest=configuration,
    )

    assert report["ready"] is False
    assert "artifact_hashes_complete" in report["failed_checks"]


def test_readiness_rejects_missing_artifact_or_invalid_runtime_fingerprint(
    tmp_path: Path,
) -> None:
    calibration = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )
    configuration = _configuration_manifest(sealed=True)
    configuration["artifacts"].pop("index_manifest")
    configuration["configuration_fingerprint"] = "not-a-sha256"

    report = assess_final_readiness(
        dataset_manifest=_dataset_manifest(ready=True),
        judge_report=_judge_report(passing=True),
        selected_judge_model="judge-model",
        calibration_report=calibration,
        configuration_manifest=configuration,
    )

    assert report["ready"] is False
    assert "artifact_hashes_complete" in report["failed_checks"]
    assert "runtime_configuration_sealed" in report["failed_checks"]


def test_readiness_rejects_partial_human_and_judge_evidence(tmp_path: Path) -> None:
    calibration = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )
    dataset = _dataset_manifest(ready=True)
    dataset["annotation_summary"]["double_labeled_cases"] = 399
    judge = _judge_report(passing=True)
    judge["models"]["judge-model"]["splits"]["judge_validation"]["summary"][
        "evaluated_predictions"
    ] = 199

    report = assess_final_readiness(
        dataset_manifest=dataset,
        judge_report=judge,
        selected_judge_model="judge-model",
        calibration_report=calibration,
        configuration_manifest=_configuration_manifest(sealed=True),
    )

    assert report["ready"] is False
    assert "holdout_annotations_sealed" in report["failed_checks"]
    assert "judge_validation_complete" in report["failed_checks"]


def test_write_calibration_evidence_creates_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / "final_calibration.json"
    output_markdown = tmp_path / "final_calibration.md"

    report = write_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert json.loads(output_json.read_text()) == report
    assert "391/400" in output_markdown.read_text()


def test_seal_runtime_configuration_hashes_every_required_artifact(tmp_path: Path) -> None:
    calibration_path = _write_json(
        tmp_path / "calibration.json",
        build_calibration_evidence(
            deterministic_path=_write_json(
                tmp_path / "deterministic.json", _deterministic_report()
            ),
            model_path=_write_json(tmp_path / "model.json", _model_report()),
        ),
    )
    dataset_manifest = _write_json(
        tmp_path / "dataset_manifest.json",
        {**_dataset_manifest(ready=True), "holdout_frozen": True},
    )
    policy = tmp_path / "policy.toml"
    policy.write_text("[retrieval]\nallowed_visibility = [\"public\"]\n")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id":"course-policy"}\n')
    index_manifest = _write_json(
        tmp_path / "index_manifest.json",
        {"embedding_model": "BAAI/bge-m3"},
    )
    output = tmp_path / "configuration_freeze.json"

    sealed = seal_runtime_configuration(
        dataset_manifest_path=dataset_manifest,
        calibration_report_path=calibration_path,
        policy_path=policy,
        corpus_path=corpus,
        index_manifest_path=index_manifest,
        output_path=output,
    )

    assert sealed["sealed"] is True
    assert sealed["holdout_used"] is False
    assert len(sealed["configuration_fingerprint"]) == 64
    assert set(sealed["artifacts"]) == {
        "dataset_manifest",
        "calibration_evidence",
        "policy",
        "corpus",
        "index_manifest",
    }
    assert all(len(item["sha256"]) == 64 for item in sealed["artifacts"].values())
    assert json.loads(output.read_text()) == sealed


def test_seal_runtime_configuration_rejects_unfrozen_or_holdout_evidence(
    tmp_path: Path,
) -> None:
    dataset_manifest = _dataset_manifest(ready=True)
    dataset_manifest["holdout_frozen"] = False
    calibration = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )
    calibration_path = _write_json(tmp_path / "calibration.json", calibration)
    paths = []
    for name in ("policy.toml", "corpus.jsonl", "index.json"):
        path = tmp_path / name
        path.write_text(name)
        paths.append(path)

    with pytest.raises(FinalEvidenceError, match="frozen"):
        seal_runtime_configuration(
            dataset_manifest_path=_write_json(
                tmp_path / "dataset_manifest.json", dataset_manifest
            ),
            calibration_report_path=calibration_path,
            policy_path=paths[0],
            corpus_path=paths[1],
            index_manifest_path=paths[2],
            output_path=tmp_path / "freeze.json",
        )


def test_seal_runtime_configuration_rejects_partial_double_review(tmp_path: Path) -> None:
    dataset = _dataset_manifest(ready=True)
    dataset["annotation_summary"]["double_labeled_cases"] = 399
    calibration_path = _write_json(
        tmp_path / "calibration.json",
        build_calibration_evidence(
            deterministic_path=_write_json(
                tmp_path / "deterministic.json", _deterministic_report()
            ),
            model_path=_write_json(tmp_path / "model.json", _model_report()),
        ),
    )
    paths = []
    for name in ("policy.toml", "corpus.jsonl", "index.json"):
        path = tmp_path / name
        path.write_text(name)
        paths.append(path)

    with pytest.raises(FinalEvidenceError, match="reviewed and sealed"):
        seal_runtime_configuration(
            dataset_manifest_path=_write_json(tmp_path / "dataset.json", dataset),
            calibration_report_path=calibration_path,
            policy_path=paths[0],
            corpus_path=paths[1],
            index_manifest_path=paths[2],
            output_path=tmp_path / "freeze.json",
        )


def test_build_final_evidence_cli_writes_both_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "final.json"
    output_markdown = tmp_path / "final.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "build-final-evidence",
            "--deterministic-report",
            str(_write_json(tmp_path / "deterministic.json", _deterministic_report())),
            "--model-report",
            str(_write_json(tmp_path / "model.json", _model_report())),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["cases"] == 400
    assert output_json.exists()
    assert output_markdown.exists()


def test_seal_final_config_cli_writes_configuration_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibration_path = _write_json(
        tmp_path / "calibration.json",
        build_calibration_evidence(
            deterministic_path=_write_json(
                tmp_path / "deterministic.json", _deterministic_report()
            ),
            model_path=_write_json(tmp_path / "model.json", _model_report()),
        ),
    )
    dataset_manifest = _write_json(
        tmp_path / "dataset_manifest.json", _dataset_manifest(ready=True)
    )
    policy = tmp_path / "policy.toml"
    policy.write_text("[retrieval]\nallowed_visibility = [\"public\"]\n")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id":"course-policy"}\n')
    index_manifest = _write_json(
        tmp_path / "index_manifest.json", {"embedding_model": "BAAI/bge-m3"}
    )
    output = tmp_path / "configuration_freeze.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "seal-final-config",
            "--dataset-manifest",
            str(dataset_manifest),
            "--calibration-report",
            str(calibration_path),
            "--policy",
            str(policy),
            "--course-corpus",
            str(corpus),
            "--index-manifest",
            str(index_manifest),
            "--output-json",
            str(output),
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["sealed"] is True
    assert json.loads(output.read_text()) == payload


def test_check_final_readiness_cli_exits_nonzero_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibration = build_calibration_evidence(
        deterministic_path=_write_json(tmp_path / "deterministic.json", _deterministic_report()),
        model_path=_write_json(tmp_path / "model.json", _model_report()),
    )
    output = tmp_path / "readiness.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "check-final-readiness",
            "--dataset-manifest",
            str(_write_json(tmp_path / "dataset.json", _dataset_manifest(ready=False))),
            "--judge-report",
            str(_write_json(tmp_path / "judge.json", _judge_report(passing=False))),
            "--selected-judge-model",
            "judge-model",
            "--calibration-report",
            str(_write_json(tmp_path / "calibration.json", calibration)),
            "--configuration-manifest",
            str(
                _write_json(
                    tmp_path / "configuration.json",
                    _configuration_manifest(sealed=False),
                )
            ),
            "--output-json",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert json.loads(output.read_text()) == payload
