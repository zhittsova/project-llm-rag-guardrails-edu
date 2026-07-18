import json
import re
from copy import deepcopy
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import pytest

from guardrails_llm.dispositions import ResponseDisposition
from guardrails_llm.evaluation import EvalCase, load_eval_cases
from guardrails_llm.evaluation_dataset import (
    DATASET_FILENAMES,
    GUARDRAIL_FAMILIES,
    TOPICS,
    DatasetValidationError,
    apply_holdout_annotations,
    cohens_kappa,
    finalize_holdout_annotations,
    generate_evaluation_dataset,
    load_and_validate_evaluation_dataset,
    load_evaluation_cases_for_run,
    validate_evaluation_dataset,
    verify_dataset_split_manifest,
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
    assert Counter(next(iter(splits)) for splits in parents.values()) == {
        "development": 15,
        "calibration": 5,
        "holdout": 5,
    }


def test_german_cases_use_german_topic_names() -> None:
    german_topics = {slug: topic_de for slug, _topic, topic_de, _doc, _claim in TOPICS}

    for cases in _cases_by_split().values():
        for case in cases:
            if case.language == "de":
                slug = case.parent_case_id.removeprefix("m3v2-")
                assert german_topics[slug].casefold() in case.question.casefold()


def test_prompt_templates_do_not_cross_split_boundaries() -> None:
    topic_names = [name for _slug, en, de, _doc, _claim in TOPICS for name in (en, de)]
    contexts = [
        "lecture review",
        "lab discussion",
        "deployment demo",
        "evaluation report",
        "course-policy workshop",
        "Vorlesungsrückblick",
        "Labordiskussion",
        "Deployment-Demo",
        "Evaluationsbericht",
        "Workshop zur Kursrichtlinie",
    ]

    def template(question: str) -> str:
        normalized = question.casefold()
        for value in sorted(topic_names, key=len, reverse=True):
            normalized = normalized.replace(value.casefold(), "<topic>")
        for value in sorted(contexts, key=len, reverse=True):
            normalized = normalized.replace(value.casefold(), "<context>")
        return " ".join(normalized.split())

    templates = {
        split: {template(case.question) for case in cases}
        for split, cases in _cases_by_split().items()
    }
    assert templates["development"].isdisjoint(templates["calibration"])
    assert templates["development"].isdisjoint(templates["holdout"])
    assert templates["calibration"].isdisjoint(templates["holdout"])


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


def test_higher_order_functions_reference_the_functions_as_objects_lecture() -> None:
    cases = [
        case
        for split in _cases_by_split().values()
        for case in split
        if case.parent_case_id == "m3v2-higher-order"
        and case.family_id == "course_qa"
    ]

    assert cases
    assert all(case.expected_doc_ids == ["lec08"] for case in cases)


def test_benign_near_miss_evidence_matches_supported_policy_claims() -> None:
    cases = [
        case
        for split in _cases_by_split().values()
        for case in split
        if case.coverage_role == "benign_near_miss"
    ]

    expected_claims = {
        "prompt_injection": "user attempts to override instructions",
        "pii": "access personal student records",
        "unsafe_request": "malicious and harmful requests",
        "academic_integrity": "conceptual explanations for algorithms",
        "unsupported": "must not invent fictional tasks",
        "groundedness": "must not invent fictional tasks",
        "metadata_access": "access personal student records",
    }
    corpus = {
        row["doc_id"]: row["text"].casefold()
        for row in map(
            json.loads,
            Path("data/python_course_docs.jsonl").read_text().splitlines(),
        )
    }

    for case in cases:
        assert case.required_claims == [expected_claims[case.family_id]]
        evidence = " ".join(corpus[doc_id] for doc_id in case.expected_doc_ids or [])
        assert expected_claims[case.family_id].casefold() in evidence


def test_course_qa_required_claims_are_present_in_referenced_lectures() -> None:
    corpus = {
        row["doc_id"]: row["text"]
        for row in map(
            json.loads,
            Path("data/python_course_docs.jsonl").read_text().splitlines(),
        )
    }
    cases = [
        case
        for split in _cases_by_split().values()
        for case in split
        if case.family_id == "course_qa"
    ]

    for case in cases:
        evidence = " ".join(corpus[doc_id] for doc_id in case.expected_doc_ids or [])
        normalized_evidence = re.sub(r"[^a-z0-9]+", " ", evidence.casefold()).strip()
        for claim in case.required_claims or []:
            normalized_claim = re.sub(r"[^a-z0-9]+", " ", claim.casefold()).strip()
            assert normalized_claim in normalized_evidence, case.case_id


def test_boundary_control_is_not_used_for_redirect_attacks() -> None:
    redirects = [
        case
        for cases in _cases_by_split().values()
        for case in cases
        if case.resolved_expected_behavior() is ResponseDisposition.REDIRECT
    ]

    assert all(case.coverage_role != "boundary_control" for case in redirects)


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


def test_tuning_split_unsupported_cases_do_not_overlap_with_pii() -> None:
    cases_by_split = _cases_by_split()

    unsupported = [
        case
        for split in ("development", "calibration")
        for case in cases_by_split[split]
        if case.family_id == "unsupported"
    ]

    assert unsupported
    assert all("private api" not in case.question.casefold() for case in unsupported)
    assert all("privaten api" not in case.question.casefold() for case in unsupported)


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


def test_dataset_split_manifest_binds_exact_versioned_file(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path, replace_frozen_holdout=True)
    manifest_path = tmp_path / DATASET_FILENAMES["manifest"]
    development_path = tmp_path / DATASET_FILENAMES["development"]

    evidence = verify_dataset_split_manifest(
        manifest_path,
        split="development",
        split_path=development_path,
    )

    assert evidence["dataset_version"] == "milestone3-v2"
    assert len(evidence["dataset_manifest_sha256"]) == 64
    assert evidence["split_sha256"] == json.loads(
        manifest_path.read_text(encoding="utf-8")
    )["files"]["development"]["sha256"]


def test_dataset_split_manifest_rejects_relabelled_copy(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path, replace_frozen_holdout=True)
    manifest_path = tmp_path / DATASET_FILENAMES["manifest"]
    development_path = tmp_path / DATASET_FILENAMES["development"]
    copied_path = tmp_path / "copied-development.jsonl"
    copied_path.write_bytes(development_path.read_bytes())

    with pytest.raises(DatasetValidationError, match="exact versioned file"):
        verify_dataset_split_manifest(
            manifest_path,
            split="development",
            split_path=copied_path,
        )


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


def test_writer_updates_tuning_splits_without_replacing_manifest_bound_holdout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_evaluation_dataset(tmp_path)
    holdout_path = tmp_path / DATASET_FILENAMES["holdout"]
    holdout_before = holdout_path.read_bytes()
    original_generate = generate_evaluation_dataset

    def changed_generator():
        rows = deepcopy(original_generate())
        rows["development"][0]["question"] += " Tuning revision."
        rows["holdout"][0]["question"] += " Generator drift."
        return rows

    monkeypatch.setattr(
        "guardrails_llm.evaluation_dataset.generate_evaluation_dataset",
        changed_generator,
    )

    manifest = write_evaluation_dataset(tmp_path)

    development = load_eval_cases(tmp_path / DATASET_FILENAMES["development"])
    assert development[0].question.endswith("Tuning revision.")
    assert holdout_path.read_bytes() == holdout_before
    assert manifest["files"]["holdout"]["sha256"] == sha256(holdout_before).hexdigest()


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
            "annotator_a_expected_doc_ids": holdout[0].expected_doc_ids,
            "annotator_b_expected_doc_ids": holdout[0].expected_doc_ids,
            "adjudicated_expected_doc_ids": holdout[0].expected_doc_ids,
            "annotator_a_required_claims": holdout[0].required_claims,
            "annotator_b_required_claims": holdout[0].required_claims,
            "adjudicated_required_claims": holdout[0].required_claims,
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
            "annotator_a_expected_doc_ids": holdout[0].expected_doc_ids,
            "annotator_b_expected_doc_ids": holdout[0].expected_doc_ids,
            "adjudicated_expected_doc_ids": holdout[0].expected_doc_ids,
            "annotator_a_required_claims": holdout[0].required_claims,
            "annotator_b_required_claims": holdout[0].required_claims,
            "adjudicated_required_claims": holdout[0].required_claims,
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
            "annotator_a_expected_doc_ids": [],
            "annotator_b_expected_doc_ids": holdout[1].expected_doc_ids,
            "adjudicated_expected_doc_ids": [],
            "annotator_a_required_claims": [],
            "annotator_b_required_claims": holdout[1].required_claims,
            "adjudicated_required_claims": [],
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
        "expected_doc_ids_exact_agreement": 0.5,
        "required_claims_exact_agreement": 0.5,
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


def _complete_holdout_annotations(tmp_path: Path) -> None:
    holdout = {
        case.case_id: case
        for case in load_eval_cases(tmp_path / DATASET_FILENAMES["holdout"])
    }
    annotation_path = tmp_path / DATASET_FILENAMES["annotations"]
    annotations = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    for index, row in enumerate(annotations):
        case = holdout[row["case_id"]]
        behavior = case.resolved_expected_behavior().value
        evidence = case.evidence_available
        expected_doc_ids = case.expected_doc_ids
        required_claims = case.required_claims
        if index == 0:
            behavior = "block"
            evidence = False
            expected_doc_ids = []
            required_claims = []
        row.update(
            {
                "annotator_a_id": "reviewer-1",
                "annotator_b_id": "reviewer-2",
                "adjudicator_id": "reviewer-1",
                "annotator_a_behavior": behavior,
                "annotator_b_behavior": behavior,
                "adjudicated_behavior": behavior,
                "annotator_a_evidence_available": evidence,
                "annotator_b_evidence_available": evidence,
                "adjudicated_evidence_available": evidence,
                "annotator_a_expected_doc_ids": expected_doc_ids,
                "annotator_b_expected_doc_ids": expected_doc_ids,
                "adjudicated_expected_doc_ids": expected_doc_ids,
                "annotator_a_required_claims": required_claims,
                "annotator_b_required_claims": required_claims,
                "adjudicated_required_claims": required_claims,
            }
        )
    annotation_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in annotations)
    )


def test_unsealed_holdout_cannot_run_even_after_review(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    _complete_holdout_annotations(tmp_path)

    with pytest.raises(DatasetValidationError, match="sealed"):
        load_evaluation_cases_for_run(
            tmp_path / DATASET_FILENAMES["holdout"],
        )


def test_finalized_holdout_uses_adjudicated_labels(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    _complete_holdout_annotations(tmp_path)

    manifest = finalize_holdout_annotations(tmp_path)
    cases = load_evaluation_cases_for_run(tmp_path / DATASET_FILENAMES["holdout"])

    assert manifest["annotation_sealed"] is True
    assert manifest["holdout_review_status"] == "adjudicated"
    assert cases[0].annotation_status == "adjudicated"
    assert cases[0].resolved_expected_behavior() is ResponseDisposition.BLOCK
    assert cases[0].evidence_available is False
    assert cases[0].expected_doc_ids == []
    assert cases[0].required_claims == []


def test_sealed_annotation_tampering_is_detected(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    _complete_holdout_annotations(tmp_path)
    finalize_holdout_annotations(tmp_path)
    annotation_path = tmp_path / DATASET_FILENAMES["annotations"]
    annotation_path.write_text(annotation_path.read_text() + "\n")

    with pytest.raises(DatasetValidationError, match="annotation.*SHA-256"):
        load_evaluation_cases_for_run(tmp_path / DATASET_FILENAMES["holdout"])


def test_finalization_rejects_inconsistent_adjudicated_evidence(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    _complete_holdout_annotations(tmp_path)
    holdout = {
        case.case_id: case
        for case in load_eval_cases(tmp_path / DATASET_FILENAMES["holdout"])
    }
    annotation_path = tmp_path / DATASET_FILENAMES["annotations"]
    annotations = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    target = next(
        row
        for row in annotations
        if holdout[row["case_id"]].resolved_expected_behavior() is ResponseDisposition.BLOCK
    )
    target.update(
        {
            "annotator_a_behavior": "answer",
            "annotator_b_behavior": "answer",
            "adjudicated_behavior": "answer",
            "annotator_a_evidence_available": True,
            "annotator_b_evidence_available": True,
            "adjudicated_evidence_available": True,
            "annotator_a_expected_doc_ids": [],
            "annotator_b_expected_doc_ids": [],
            "adjudicated_expected_doc_ids": [],
            "annotator_a_required_claims": [],
            "annotator_b_required_claims": [],
            "adjudicated_required_claims": [],
        }
    )
    annotation_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in annotations)
    )

    with pytest.raises(DatasetValidationError, match="expected evidence metadata"):
        finalize_holdout_annotations(tmp_path)
