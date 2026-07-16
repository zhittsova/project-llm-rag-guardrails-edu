from pathlib import Path

from guardrails_llm.evaluation import EvalCase, load_eval_cases, select_eval_split


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
