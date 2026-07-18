import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from guardrails_llm.dispositions import ResponseDisposition
from guardrails_llm.evaluation import EvalCase
from guardrails_llm.evaluation_dataset import (
    DATASET_FILENAMES,
    GUARDRAIL_FAMILIES,
    DatasetValidationError,
    apply_holdout_annotations,
    cohens_kappa,
    generate_evaluation_dataset,
    load_and_validate_evaluation_dataset,
    validate_evaluation_dataset,
    write_evaluation_dataset,
)


def _cases_by_split() -> dict[str, list[EvalCase]]:
    return {
        split: [EvalCase(**row) for row in rows]
        for split, rows in generate_evaluation_dataset().items()
    }


def test_generated_dataset_has_frozen_split_and_action_counts() -> None:
    cases_by_split = _cases_by_split()

    assert {split: len(cases) for split, cases in cases_by_split.items()} == {
        "development": 1200,
        "calibration": 400,
        "holdout": 400,
    }
    dispositions = Counter(
        case.resolved_expected_behavior()
        for cases in cases_by_split.values()
        for case in cases
    )
    assert dispositions == {
        ResponseDisposition.ANSWER: 500,
        ResponseDisposition.BLOCK: 500,
        ResponseDisposition.ABSTAIN: 500,
        ResponseDisposition.REDIRECT: 500,
    }
    assert Counter(case.language for cases in cases_by_split.values() for case in cases) == {
        "en": 1500,
        "de": 500,
    }


def test_generated_dataset_keeps_parent_families_inside_one_split() -> None:
    parents: dict[str, set[str]] = defaultdict(set)
    for split, cases in _cases_by_split().items():
        for case in cases:
            parents[case.parent_case_id].add(split)

    assert all(len(splits) == 1 for splits in parents.values())


def test_generated_dataset_covers_each_guardrail_family_in_every_split() -> None:
    cases_by_split = _cases_by_split()

    for cases in cases_by_split.values():
        roles: dict[str, set[str]] = defaultdict(set)
        for case in cases:
            roles[case.family_id].add(case.coverage_role)
        for family in GUARDRAIL_FAMILIES:
            family_roles = roles[family]
            assert {
                "positive_direct",
                "positive_variant",
                "benign_near_miss",
            } <= family_roles


def test_generated_answers_cover_course_qa_and_guardrail_near_misses() -> None:
    answers = [
        case
        for cases in _cases_by_split().values()
        for case in cases
        if case.resolved_expected_behavior() is ResponseDisposition.ANSWER
    ]

    assert Counter(case.family_id for case in answers)["course_qa"] == 250
    assert sum(case.coverage_role == "benign_near_miss" for case in answers) == 250


def test_generated_evidence_references_real_python_course_documents() -> None:
    corpus_doc_ids = {
        json.loads(line)["doc_id"]
        for line in Path("data/python_course_docs.jsonl").read_text().splitlines()
    }

    for cases in _cases_by_split().values():
        for case in cases:
            assert set(case.expected_doc_ids or []) <= corpus_doc_ids


def test_validator_rejects_unknown_expected_document_id() -> None:
    cases_by_split = _cases_by_split()
    original = cases_by_split["development"][0]
    cases_by_split["development"][0] = EvalCase(
        **{**original.__dict__, "expected_doc_ids": ["missing-document"]}
    )

    with pytest.raises(DatasetValidationError, match="unknown expected_doc_id"):
        validate_evaluation_dataset(cases_by_split, known_doc_ids={"lec01"})


def test_generated_dataset_has_no_duplicate_questions_or_case_ids() -> None:
    cases = [case for split in _cases_by_split().values() for case in split]

    assert len({case.case_id for case in cases}) == len(cases)
    assert len({case.question.casefold() for case in cases}) == len(cases)


def test_holdout_is_blocked_until_independent_review() -> None:
    cases_by_split = _cases_by_split()

    summary = validate_evaluation_dataset(cases_by_split)
    assert summary["holdout_review_status"] == "pending_double_review"

    with pytest.raises(DatasetValidationError, match="400 holdout cases.*adjudicated"):
        validate_evaluation_dataset(cases_by_split, require_reviewed_holdout=True)


def test_validator_rejects_parent_leakage() -> None:
    cases_by_split = _cases_by_split()
    leaked = cases_by_split["holdout"][0]
    cases_by_split["development"][0] = EvalCase(
        **(
            {
                **leaked.__dict__,
                "case_id": "leaked-parent-case",
                "question": f"{leaked.question} Cross-split copy.",
                "split": "development",
            }
        )
    )

    with pytest.raises(DatasetValidationError, match="parent_case_id.*multiple splits"):
        validate_evaluation_dataset(cases_by_split)


def test_writer_is_deterministic_and_manifest_hashes_match(tmp_path: Path) -> None:
    first = write_evaluation_dataset(tmp_path)
    first_contents = {
        name: (tmp_path / filename).read_bytes()
        for name, filename in DATASET_FILENAMES.items()
    }
    second = write_evaluation_dataset(tmp_path)
    second_contents = {
        name: (tmp_path / filename).read_bytes()
        for name, filename in DATASET_FILENAMES.items()
    }

    assert first == second
    assert first_contents == second_contents
    manifest = json.loads((tmp_path / DATASET_FILENAMES["manifest"]).read_text())
    assert manifest["schema_version"] == 2
    assert manifest["total_cases"] == 2000
    assert manifest["holdout_frozen"] is True
    assert manifest["holdout_review_status"] == "pending_double_review"


def test_writer_preserves_existing_human_annotations(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    annotation_path = tmp_path / DATASET_FILENAMES["annotations"]
    rows = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    rows[0]["annotator_a_id"] = "reviewer-1"
    rows[0]["annotator_a_behavior"] = "answer"
    annotation_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )

    write_evaluation_dataset(tmp_path)

    preserved = json.loads(annotation_path.read_text().splitlines()[0])
    assert preserved["annotator_a_id"] == "reviewer-1"
    assert preserved["annotator_a_behavior"] == "answer"


def test_writer_refuses_to_replace_changed_frozen_holdout(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    holdout_path = tmp_path / DATASET_FILENAMES["holdout"]
    holdout_path.write_text(holdout_path.read_text() + "\n")

    with pytest.raises(DatasetValidationError, match="frozen holdout"):
        write_evaluation_dataset(tmp_path)


def test_cohens_kappa_reports_perfect_and_chance_adjusted_agreement() -> None:
    labels = ["answer", "block", "abstain", "redirect"]
    assert cohens_kappa(labels, labels) == 1.0
    assert cohens_kappa(
        ["answer", "answer", "block", "block"],
        ["answer", "block", "answer", "block"],
    ) == 0.0


def test_cohens_kappa_rejects_different_annotation_lengths() -> None:
    with pytest.raises(ValueError, match="same number"):
        cohens_kappa(["answer"], ["answer", "block"])


def test_holdout_annotations_require_two_distinct_reviewers() -> None:
    holdout = _cases_by_split()["holdout"][:1]
    annotations = [
        {
            "case_id": holdout[0].case_id,
            "annotator_a_id": "reviewer-1",
            "annotator_b_id": "reviewer-1",
            "adjudicator_id": "reviewer-1",
            "annotator_a_behavior": "answer",
            "annotator_b_behavior": "answer",
            "adjudicated_behavior": "answer",
            "annotator_a_evidence_available": True,
            "annotator_b_evidence_available": True,
            "adjudicated_evidence_available": True,
        }
    ]

    with pytest.raises(DatasetValidationError, match="distinct annotators"):
        apply_holdout_annotations(holdout, annotations)


def test_holdout_annotations_compute_agreement_and_apply_adjudication() -> None:
    holdout = _cases_by_split()["holdout"][:2]
    annotations = [
        {
            "case_id": holdout[0].case_id,
            "annotator_a_id": "reviewer-1",
            "annotator_b_id": "reviewer-2",
            "adjudicator_id": "reviewer-1",
            "annotator_a_behavior": "answer",
            "annotator_b_behavior": "answer",
            "adjudicated_behavior": "answer",
            "annotator_a_evidence_available": True,
            "annotator_b_evidence_available": True,
            "adjudicated_evidence_available": True,
        },
        {
            "case_id": holdout[1].case_id,
            "annotator_a_id": "reviewer-1",
            "annotator_b_id": "reviewer-2",
            "adjudicator_id": "reviewer-1",
            "annotator_a_behavior": "block",
            "annotator_b_behavior": "answer",
            "adjudicated_behavior": "block",
            "annotator_a_evidence_available": False,
            "annotator_b_evidence_available": True,
            "adjudicated_evidence_available": False,
        },
    ]

    updated, summary = apply_holdout_annotations(holdout, annotations)

    assert all(case.annotation_status == "adjudicated" for case in updated)
    assert updated[1].adjudicated_label is ResponseDisposition.BLOCK
    assert updated[1].evidence_available is False
    assert summary == {
        "total_cases": 2,
        "double_labeled_cases": 2,
        "adjudicated_cases": 2,
        "behavior_kappa": 0.0,
        "evidence_kappa": 0.0,
        "ready_for_final_holdout": True,
    }


def test_partial_adjudication_is_rejected() -> None:
    holdout = _cases_by_split()["holdout"][:1]
    annotations = [
        {
            "case_id": holdout[0].case_id,
            "annotator_a_id": "reviewer-1",
            "annotator_b_id": "reviewer-2",
            "adjudicator_id": None,
            "annotator_a_behavior": "block",
            "annotator_b_behavior": "answer",
            "adjudicated_behavior": "block",
            "annotator_a_evidence_available": False,
            "annotator_b_evidence_available": True,
            "adjudicated_evidence_available": None,
        }
    ]

    with pytest.raises(DatasetValidationError, match="complete adjudication"):
        apply_holdout_annotations(holdout, annotations)


def test_adjudication_requires_an_identified_adjudicator() -> None:
    holdout = _cases_by_split()["holdout"][:1]
    annotations = [
        {
            "case_id": holdout[0].case_id,
            "annotator_a_id": "reviewer-1",
            "annotator_b_id": "reviewer-2",
            "adjudicator_id": None,
            "annotator_a_behavior": "answer",
            "annotator_b_behavior": "answer",
            "adjudicated_behavior": "answer",
            "annotator_a_evidence_available": True,
            "annotator_b_evidence_available": True,
            "adjudicated_evidence_available": True,
        }
    ]

    with pytest.raises(DatasetValidationError, match="adjudicator ID"):
        apply_holdout_annotations(holdout, annotations)


def test_file_validator_detects_modified_frozen_holdout(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    holdout_path = tmp_path / DATASET_FILENAMES["holdout"]
    holdout_path.write_text(holdout_path.read_text() + "\n")

    with pytest.raises(DatasetValidationError, match="holdout.*SHA-256"):
        load_and_validate_evaluation_dataset(tmp_path)
