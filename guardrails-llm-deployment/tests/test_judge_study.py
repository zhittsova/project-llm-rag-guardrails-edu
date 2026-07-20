from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails_llm.evaluation import load_eval_cases
from guardrails_llm.judge_study import (
    JUDGE_SPLITS,
    finalize_human_ground_truth,
    evaluate_judge_study_models,
    judge_study_status,
    prepare_judge_study,
    reconcile_human_annotations,
    validate_annotation_file,
)
from guardrails_llm.model_calibration import load_judge_calibration_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_CASES = PROJECT_ROOT / "data" / "eval_cases_milestone3_v2_calibration.jsonl"


def test_prepare_judge_study_builds_balanced_family_disjoint_sets(
    tmp_path: Path,
) -> None:
    source_results = tmp_path / "source_results.json"
    source_results.write_text(
        json.dumps(_source_results()),
        encoding="utf-8",
    )

    manifest = prepare_judge_study(
        cases_path=CALIBRATION_CASES,
        source_results_path=source_results,
        output_dir=tmp_path / "study",
    )

    calibration = manifest["splits"]["judge_calibration"]
    validation = manifest["splits"]["judge_validation"]
    for split in (calibration, validation):
        assert split["items"] == 200
        assert split["expected_behavior_counts"] == {
            "abstain": 50,
            "answer": 50,
            "block": 50,
            "redirect": 50,
        }
        assert split["language_counts"] == {"de": 50, "en": 150}
        assert split["evidence_counts"] == {"false": 100, "true": 100}
        assert split["outcome_counts"]["correct"] > 0
        assert split["outcome_counts"]["incorrect"] > 0
    assert set(calibration["parent_case_ids"]).isdisjoint(
        validation["parent_case_ids"]
    )

    for split in JUDGE_SPLITS:
        items = _read_jsonl(tmp_path / "study" / f"{split}_items.jsonl")
        template = _read_jsonl(
            tmp_path / "study" / f"{split}_reviewer_a.jsonl"
        )
        assert len(items) == len(template) == 200
        assert "scenario" not in items[0]
        assert template[0]["grounded"] is None
        assert template[0]["annotator_id"] == ""
        visible_payloads = [
            json.dumps(
                {key: value for key, value in item.items() if key != "item_id"},
                sort_keys=True,
            )
            for item in items
        ]
        assert len(visible_payloads) == len(set(visible_payloads))

    calibration_items = _read_jsonl(
        tmp_path / "study" / "judge_calibration_items.jsonl"
    )
    validation_items = _read_jsonl(
        tmp_path / "study" / "judge_validation_items.jsonl"
    )
    assert len({item["question"] for item in calibration_items}) == 160
    assert len({item["question"] for item in validation_items}) == 200


def test_annotation_validation_reports_incomplete_templates(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    payload = _annotation("judge_calibration-item", "")
    payload["grounded"] = None
    _write_jsonl(path, [payload])

    _annotations, summary = validate_annotation_file(
        path,
        expected_item_ids={"judge_calibration-item"},
        complete=False,
    )

    assert summary["completed"] == 0
    assert summary["remaining"] == 1
    assert summary["complete"] is False


def test_complete_human_annotation_allows_empty_optional_rationale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.jsonl"
    payload = _annotation("judge_calibration-item", "reviewer-a")
    payload["rationale"] = ""
    _write_jsonl(path, [payload])

    _annotations, summary = validate_annotation_file(
        path,
        expected_item_ids={"judge_calibration-item"},
        complete=True,
    )

    assert summary["complete"] is True


def test_reconciliation_writes_only_human_disagreements(tmp_path: Path) -> None:
    item_paths = []
    reviewer_a_paths = []
    reviewer_b_paths = []
    for split in JUDGE_SPLITS:
        item_path = tmp_path / f"{split}_items.jsonl"
        reviewer_a_path = tmp_path / f"{split}_reviewer_a.jsonl"
        reviewer_b_path = tmp_path / f"{split}_reviewer_b.jsonl"
        item_id = f"{split}-item"
        _write_jsonl(item_path, [{"item_id": item_id}])
        _write_jsonl(reviewer_a_path, [_annotation(item_id, "reviewer-a")])
        reviewer_b = _annotation(item_id, "reviewer-b")
        if split == "judge_validation":
            reviewer_b["grounded"] = False
        _write_jsonl(reviewer_b_path, [reviewer_b])
        item_paths.append(item_path)
        reviewer_a_paths.append(reviewer_a_path)
        reviewer_b_paths.append(reviewer_b_path)

    report = reconcile_human_annotations(
        items_paths=item_paths,
        reviewer_a_paths=reviewer_a_paths,
        reviewer_b_paths=reviewer_b_paths,
        disagreements_output=tmp_path / "disagreements.jsonl",
        report_output=tmp_path / "report.json",
    )

    disagreements = _read_jsonl(tmp_path / "disagreements.jsonl")
    assert report["items"] == 2
    assert report["exact_five_dimension_agreements"] == 1
    assert report["items_requiring_adjudication"] == 1
    assert report["human_ground_truth_ready"] is False
    assert disagreements[0]["differing_dimensions"] == ["grounded"]
    assert disagreements[0]["grounded"] is None


def test_status_keeps_ground_truth_pending_until_adjudication_is_finalized(
    tmp_path: Path,
) -> None:
    for split in JUDGE_SPLITS:
        item_id = f"{split}-item"
        _write_jsonl(tmp_path / f"{split}_items.jsonl", [{"item_id": item_id}])
        _write_jsonl(
            tmp_path / f"{split}_reviewer_a.jsonl",
            [_annotation(item_id, "reviewer-a")],
        )
        reviewer_b = _annotation(item_id, "reviewer-b")
        if split == "judge_validation":
            reviewer_b["grounded"] = False
        _write_jsonl(tmp_path / f"{split}_reviewer_b.jsonl", [reviewer_b])

    reconcile_human_annotations(
        items_paths=[tmp_path / f"{split}_items.jsonl" for split in JUDGE_SPLITS],
        reviewer_a_paths=[
            tmp_path / f"{split}_reviewer_a.jsonl" for split in JUDGE_SPLITS
        ],
        reviewer_b_paths=[
            tmp_path / f"{split}_reviewer_b.jsonl" for split in JUDGE_SPLITS
        ],
        disagreements_output=tmp_path / "judge_disagreements.jsonl",
        report_output=tmp_path / "judge_human_agreement.json",
    )

    report = judge_study_status(tmp_path)

    assert report["reviewer_files_complete"] is True
    assert report["reconciliation_complete"] is False
    assert report["adjudications"] == {"total": 1, "completed": 0, "remaining": 1}
    assert report["human_ground_truth_ready"] is False


def test_status_rejects_stale_disagreement_file(tmp_path: Path) -> None:
    for split in JUDGE_SPLITS:
        item_id = f"{split}-item"
        _write_jsonl(tmp_path / f"{split}_items.jsonl", [{"item_id": item_id}])
        _write_jsonl(
            tmp_path / f"{split}_reviewer_a.jsonl",
            [_annotation(item_id, "reviewer-a")],
        )
        reviewer_b = _annotation(item_id, "reviewer-b")
        if split == "judge_validation":
            reviewer_b["grounded"] = False
        _write_jsonl(tmp_path / f"{split}_reviewer_b.jsonl", [reviewer_b])
    _write_jsonl(tmp_path / "judge_disagreements.jsonl", [])

    with pytest.raises(ValueError, match="disagreement file is stale"):
        judge_study_status(tmp_path)


def test_finalize_compiles_consensus_and_adjudicated_labels(tmp_path: Path) -> None:
    source_results = tmp_path / "source_results.json"
    source_results.write_text(json.dumps(_source_results()), encoding="utf-8")
    study_dir = tmp_path / "study"
    prepare_judge_study(
        cases_path=CALIBRATION_CASES,
        source_results_path=source_results,
        output_dir=study_dir,
    )
    first_item_id = None
    for split in JUDGE_SPLITS:
        for reviewer, annotator_id in (
            ("reviewer_a", "reviewer-a"),
            ("reviewer_b", "reviewer-b"),
        ):
            path = study_dir / f"{split}_{reviewer}.jsonl"
            payloads = _read_jsonl(path)
            for payload in payloads:
                payload.update(_annotation(str(payload["item_id"]), annotator_id))
            if split == "judge_calibration" and reviewer == "reviewer_b":
                first_item_id = str(payloads[0]["item_id"])
                payloads[0]["grounded"] = False
            _write_jsonl(path, payloads)

    reconcile_human_annotations(
        items_paths=[study_dir / f"{split}_items.jsonl" for split in JUDGE_SPLITS],
        reviewer_a_paths=[study_dir / f"{split}_reviewer_a.jsonl" for split in JUDGE_SPLITS],
        reviewer_b_paths=[study_dir / f"{split}_reviewer_b.jsonl" for split in JUDGE_SPLITS],
        disagreements_output=study_dir / "judge_disagreements.jsonl",
        report_output=study_dir / "judge_human_agreement.json",
    )
    disagreements = _read_jsonl(study_dir / "judge_disagreements.jsonl")
    assert disagreements[0]["item_id"] == first_item_id
    disagreements[0].update(
        {
            "adjudicator_id": "adjudicator-1",
            "grounded": True,
            "privacy_safe": True,
            "injection_safe": True,
            "integrity_safe": True,
            "refusal_appropriate": True,
            "rationale": "Evidence supports the system behavior.",
        }
    )
    _write_jsonl(study_dir / "judge_disagreements.jsonl", disagreements)

    report = finalize_human_ground_truth(study_dir=study_dir)

    calibration_labels = load_judge_calibration_cases(
        study_dir / "judge_calibration_human_ground_truth.jsonl"
    )
    validation_labels = load_judge_calibration_cases(
        study_dir / "judge_validation_human_ground_truth.jsonl"
    )
    assert report["human_ground_truth_ready"] is True
    assert report["items"] == 400
    assert report["adjudicated_items"] == 1
    assert len(calibration_labels) == len(validation_labels) == 200
    assert judge_study_status(study_dir)["human_ground_truth_ready"] is True

    prediction_paths = []
    for model in ("qwen-test", "minimax-test"):
        prediction_path = study_dir / f"{model}.jsonl"
        predictions = []
        for label in calibration_labels + validation_labels:
            predictions.append(
                {
                    "case_id": label.case_id,
                    "grounded": label.grounded,
                    "privacy_safe": label.privacy_safe,
                    "injection_safe": label.injection_safe,
                    "integrity_safe": label.integrity_safe,
                    "refusal_appropriate": label.refusal_appropriate,
                    "score": label.expected_score,
                    "error": None,
                    "notes": ["test prediction"],
                    "provider": "openai_compatible",
                    "model": model,
                    "latency_ms": 1.0,
                }
            )
        _write_jsonl(prediction_path, predictions)
        prediction_paths.append(prediction_path)

    comparison = evaluate_judge_study_models(
        study_dir=study_dir,
        prediction_paths=prediction_paths,
        output_path=study_dir / "comparison.json",
    )
    for model in ("qwen-test", "minimax-test"):
        validation = comparison["models"][model]["splits"]["judge_validation"]
        assert validation["quality_gates"]["all_passed"] is True


def _source_results() -> dict[str, list[dict[str, object]]]:
    cases = load_eval_cases(CALIBRATION_CASES)
    rotated = {
        "answer": "block",
        "block": "abstain",
        "abstain": "redirect",
        "redirect": "answer",
    }
    correct = []
    incorrect = []
    for case in cases:
        expected = case.resolved_expected_behavior().value
        base = {
            "case_id": case.case_id,
            "answer": "Example system output.",
            "triggers": [],
            "retrieved_evidence": [],
            "citations": [],
            "cited_doc_ids": [],
            "supporting_chunks": [],
            "grounding_supported": None,
            "grounding_confidence": None,
            "grounding_error": None,
            "unsupported_claims": [],
        }
        correct.append({**base, "actual_behavior": expected})
        incorrect.append({**base, "actual_behavior": rotated[expected]})
    return {"correct_scenario": correct, "incorrect_scenario": incorrect}


def _annotation(item_id: str, annotator_id: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "annotator_id": annotator_id,
        "grounded": True,
        "privacy_safe": True,
        "injection_safe": True,
        "integrity_safe": True,
        "refusal_appropriate": True,
        "rationale": "Reviewed against the rubric.",
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
