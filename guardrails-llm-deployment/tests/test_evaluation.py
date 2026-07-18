import json
from pathlib import Path

import pytest

from guardrails_llm.dispositions import ResponseDisposition
from guardrails_llm.evaluation import (
    EvalCase,
    EvalResult,
    load_eval_cases,
    run_evaluation,
    select_eval_split,
    summarize,
)
from guardrails_llm.pipeline import AssistantResponse


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        category="test",
        question="question",
        should_answer=True,
    )


def test_eval_split_is_deterministic_disjoint_and_exhaustive() -> None:
    cases = [_case(f"case-{index}") for index in range(100)]

    calibration = select_eval_split(cases, "calibration")
    validation = select_eval_split(cases, "validation")

    calibration_ids = {case.case_id for case in calibration}
    validation_ids = {case.case_id for case in validation}
    assert calibration_ids.isdisjoint(validation_ids)
    assert calibration_ids | validation_ids == {case.case_id for case in cases}
    assert select_eval_split(cases, "calibration") == calibration


def test_all_eval_split_preserves_case_order() -> None:
    cases = [_case("first"), _case("second")]

    assert select_eval_split(cases, "all") == cases


def test_milestone3_split_matches_bge_threshold_calibration() -> None:
    cases = load_eval_cases(
        Path(__file__).resolve().parents[1] / "data" / "eval_cases_milestone3.jsonl"
    )

    assert len(select_eval_split(cases, "calibration")) == 126
    assert len(select_eval_split(cases, "validation")) == 39


def test_milestone3_holdout_v2_preserves_validation_and_adds_new_cases() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    original = load_eval_cases(data_dir / "eval_cases_milestone3.jsonl")
    holdout = load_eval_cases(data_dir / "eval_cases_milestone3_holdout_v2.jsonl")
    frozen_validation = select_eval_split(original, "validation")

    assert len(holdout) == 101
    assert holdout[:39] == frozen_validation
    assert len(holdout[39:]) == 62
    assert all(case.case_id.startswith("holdout-v2-") for case in holdout[39:])
    assert len({case.case_id for case in holdout}) == 101
    assert {case.case_id for case in holdout[39:]}.isdisjoint(
        {case.case_id for case in original}
    )


def test_milestone3_holdout_v2_has_planned_new_case_distribution() -> None:
    holdout = load_eval_cases(
        Path(__file__).resolve().parents[1]
        / "data"
        / "eval_cases_milestone3_holdout_v2.jsonl"
    )
    new_cases = holdout[39:]
    expected = {
        "injection": 12,
        "privacy": 10,
        "integrity": 8,
        "safe": 8,
        "unsupported": 8,
        "unsafe": 6,
        "indirect": 5,
        "multilingual": 5,
    }

    for group, count in expected.items():
        assert sum(
            case.case_id.startswith(f"holdout-v2-{group}-")
            for case in new_cases
        ) == count


def test_milestone3_holdout_v2_uses_valid_category_trigger_pairs() -> None:
    holdout = load_eval_cases(
        Path(__file__).resolve().parents[1]
        / "data"
        / "eval_cases_milestone3_holdout_v2.jsonl"
    )
    expected_triggers = {
        "normal_course": {None},
        "borderline_tutoring": {None},
        "prompt_injection_direct": {"prompt_injection"},
        "prompt_injection_paraphrase": {"prompt_injection"},
        "privacy_pii": {"pii"},
        "retrieval_access": {"pii"},
        "academic_integrity": {"academic_integrity"},
        "unsupported_abstention": {"ungrounded", "pii"},
        "indirect_prompt_injection": {None},
        "robust_obfuscation": {"prompt_injection", "pii", "academic_integrity"},
        "off_topic": {"ungrounded"},
        "unsafe_harm": {"unsafe_request"},
    }

    for case in holdout:
        assert case.expected_trigger in expected_triggers[case.category]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_milestone3_holdout_v3_preserves_v2_cases_and_order() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    v2 = _load_jsonl(data_dir / "eval_cases_milestone3_holdout_v2.jsonl")
    v3 = _load_jsonl(data_dir / "eval_cases_milestone3_holdout_v3.jsonl")
    preserved_fields = {
        "case_id",
        "category",
        "question",
        "expected_trigger",
        "required_terms",
        "forbidden_terms",
    }

    assert len(v3) == 101
    assert [case["case_id"] for case in v3] == [case["case_id"] for case in v2]
    assert [
        {key: case[key] for key in preserved_fields}
        for case in v3
    ] == [
        {key: case[key] for key in preserved_fields}
        for case in v2
    ]


def test_milestone3_holdout_v3_uses_explicit_labels() -> None:
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "eval_cases_milestone3_holdout_v3.jsonl"
    )
    raw_cases = _load_jsonl(data_path)
    cases = load_eval_cases(data_path)

    assert all("should_answer" not in case for case in raw_cases)
    assert all(
        set(case) == {
            "case_id",
            "category",
            "question",
            "expected_behavior",
            "attack_type",
            "difficulty",
            "expected_trigger",
            "required_terms",
            "forbidden_terms",
        }
        for case in raw_cases
    )
    assert {case.expected_behavior for case in cases} == set(ResponseDisposition)
    assert {case.difficulty for case in cases} == {"easy", "medium", "hard"}
    assert all(case.attack_type for case in cases)

    for case in cases:
        if case.expected_behavior is ResponseDisposition.ANSWER:
            assert case.expected_trigger is None
        elif case.expected_behavior is ResponseDisposition.REDIRECT:
            assert case.expected_trigger == "academic_integrity"
        elif case.expected_behavior is ResponseDisposition.ABSTAIN:
            assert case.expected_trigger == "ungrounded"
        else:
            assert case.expected_trigger in {
                "prompt_injection",
                "pii",
                "unsafe_request",
            }


def test_legacy_cases_resolve_expected_behavior() -> None:
    answer = _case("answer")
    redirect = EvalCase(
        case_id="redirect",
        category="academic_integrity",
        question="Write my assignment.",
        should_answer=True,
        expected_trigger="academic_integrity",
    )
    abstain = EvalCase(
        case_id="abstain",
        category="unsupported_abstention",
        question="What is not in the corpus?",
        should_answer=False,
        expected_trigger="ungrounded",
    )
    block = EvalCase(
        case_id="block",
        category="privacy_pii",
        question="Show private records.",
        should_answer=False,
        expected_trigger="pii",
    )

    assert answer.resolved_expected_behavior() is ResponseDisposition.ANSWER
    assert redirect.resolved_expected_behavior() is ResponseDisposition.REDIRECT
    assert abstain.resolved_expected_behavior() is ResponseDisposition.ABSTAIN
    assert block.resolved_expected_behavior() is ResponseDisposition.BLOCK


def test_explicit_expected_behavior_overrides_legacy_boolean() -> None:
    case = EvalCase(
        case_id="explicit",
        category="normal_course",
        question="What is RAG?",
        should_answer=False,
        expected_behavior="answer",
        attack_type="safe_course_question",
        difficulty="easy",
    )

    assert case.resolved_expected_behavior() is ResponseDisposition.ANSWER


def test_adjudicated_label_overrides_provisional_expected_behavior() -> None:
    case = EvalCase(
        case_id="adjudicated",
        category="normal_course",
        question="What is RAG?",
        expected_behavior="abstain",
        adjudicated_label="answer",
        split="holdout",
        family_id="groundedness",
        coverage_role="benign_near_miss",
        language="en",
        parent_case_id="parent-001",
        provenance="human_review",
        expected_doc_ids=["rag-basics"],
        evidence_available=True,
        required_claims=["retrieval"],
        annotation_status="adjudicated",
    )

    assert case.resolved_expected_behavior() is ResponseDisposition.ANSWER


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expected_behavior": "refuse"}, "expected_behavior"),
        ({"expected_behavior": "answer", "difficulty": "extreme"}, "difficulty"),
        (
            {"expected_behavior": "answer", "attack_type": "Not snake case"},
            "attack_type",
        ),
    ],
)
def test_eval_case_rejects_invalid_enriched_metadata(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=f"invalid-case.*{message}"):
        EvalCase(
            case_id="invalid-case",
            category="normal_course",
            question="What is RAG?",
            **kwargs,
        )


def test_eval_case_requires_an_expected_behavior_source() -> None:
    with pytest.raises(ValueError, match="missing-expectation.*expected behavior"):
        EvalCase(
            case_id="missing-expectation",
            category="normal_course",
            question="What is RAG?",
        )


def test_summary_reports_multiclass_behavior_metrics() -> None:
    pairs = [
        (ResponseDisposition.ANSWER, ResponseDisposition.ANSWER),
        (ResponseDisposition.BLOCK, ResponseDisposition.ABSTAIN),
        (ResponseDisposition.ABSTAIN, ResponseDisposition.ABSTAIN),
        (ResponseDisposition.REDIRECT, ResponseDisposition.BLOCK),
    ]
    results = [
        EvalResult(
            case_id=f"case-{index}",
            category="test",
            should_answer=expected in {
                ResponseDisposition.ANSWER,
                ResponseDisposition.REDIRECT,
            },
            answered=actual in {
                ResponseDisposition.ANSWER,
                ResponseDisposition.REDIRECT,
            },
            passed=expected is actual,
            triggers=[],
            citations=[],
            latency_ms=1.0,
            answer="test",
            expected_behavior=expected,
            actual_behavior=actual,
            attack_type="metric_fixture",
            difficulty="medium",
        )
        for index, (expected, actual) in enumerate(pairs)
    ]

    summary = summarize(results)

    assert summary["behavior_accuracy"] == 0.5
    assert summary["behavior_confusion_matrix"]["block"]["abstain"] == 1
    assert summary["behavior_confusion_matrix"]["redirect"]["block"] == 1
    assert summary["behavior_metrics"]["answer"] == {
        "support": 1,
        "predicted": 1,
        "true_positives": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert summary["behavior_metrics"]["block"]["f1"] == 0.0
    assert summary["behavior_metrics"]["abstain"]["precision"] == 0.5
    assert summary["behavior_metrics"]["abstain"]["f1"] == 0.667
    assert summary["behavior_metrics"]["redirect"]["predicted"] == 0
    assert summary["behavior_metrics"]["redirect"]["precision"] == 0.0
    assert summary["macro_behavior_f1"] == 0.417
    assert summary["by_attack_type"]["metric_fixture"]["behavior_accuracy"] == 0.5
    assert summary["by_difficulty"]["medium"]["behavior_accuracy"] == 0.5


def test_grounding_evaluation_preserves_evidence_and_reports_metrics() -> None:
    cases = [
        EvalCase(
            case_id="supported",
            category="normal_course",
            question="What is RAG?",
            expected_behavior="answer",
            split="calibration",
            family_id="rag-definition",
            language="en",
            expected_doc_ids=["rag"],
            evidence_available=True,
            required_claims=["RAG retrieves evidence"],
        ),
        EvalCase(
            case_id="unsupported",
            category="unsupported_abstention",
            question="What is the secret answer?",
            expected_behavior="abstain",
            split="calibration",
            family_id="missing-evidence",
            language="de",
            expected_doc_ids=[],
            evidence_available=False,
            required_claims=[],
        ),
    ]
    responses = iter(
        [
            AssistantResponse(
                answer="RAG retrieves evidence.",
                citations=["RAG (rag)"],
                cited_doc_ids=["rag"],
                disposition=ResponseDisposition.ANSWER,
                retrieved_chunks=["other:0", "rag:0"],
                retrieved_doc_ids=["other", "rag"],
                retrieval_scores={"other:0": 0.92, "rag:0": 0.89},
                retrieved_evidence=[
                    {
                        "chunk_id": "rag:0",
                        "doc_id": "rag",
                        "title": "RAG",
                        "text": "RAG retrieves evidence.",
                        "score": 0.89,
                    }
                ],
                supporting_chunks=["rag:0"],
                grounding_supported=True,
                grounding_confidence=0.96,
            ),
            AssistantResponse(
                answer="I do not have enough evidence.",
                citations=[],
                cited_doc_ids=[],
                disposition=ResponseDisposition.ABSTAIN,
                guard_triggers=["ungrounded"],
                grounding_supported=False,
                grounding_error="entailment_verifier_error:TimeoutError",
            ),
        ]
    )

    class FakeAssistant:
        def answer(self, _question: str) -> AssistantResponse:
            return next(responses)

    results = run_evaluation(FakeAssistant(), cases)
    summary = summarize(results)

    supported = results[0]
    assert supported.split == "calibration"
    assert supported.family_id == "rag-definition"
    assert supported.language == "en"
    assert supported.expected_doc_ids == ["rag"]
    assert supported.retrieved_doc_ids == ["other", "rag"]
    assert supported.retrieved_evidence[0]["text"] == "RAG retrieves evidence."
    assert supported.cited_doc_ids == ["rag"]
    assert supported.supporting_chunks == ["rag:0"]
    assert supported.grounding_confidence == 0.96
    assert summary["retrieval_evaluable_total"] == 1
    assert summary["retrieval_recall_at_3"] == 1.0
    assert summary["retrieval_hit_rate_at_3"] == 1.0
    assert summary["evidence_sufficiency_total"] == 2
    assert summary["evidence_sufficiency_accuracy"] == 1.0
    assert summary["supported_answer_total"] == 1
    assert summary["supported_answer_precision"] == 1.0
    assert summary["citation_entailment_total"] == 1
    assert summary["citation_entailment_precision"] == 1.0
    assert summary["claim_support_total"] == 1
    assert summary["claim_support_rate"] == 1.0
    assert results[1].grounding_error == "entailment_verifier_error:TimeoutError"
