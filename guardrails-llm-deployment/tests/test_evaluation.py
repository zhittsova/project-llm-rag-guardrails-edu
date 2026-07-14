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
