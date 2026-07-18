from guardrails_llm.confidence_intervals import bootstrap_confidence_intervals
from guardrails_llm.evaluation import EvalResult


def _result(
    case_id: str,
    family_id: str,
    expected: str,
    actual: str,
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        category="fixture",
        should_answer=expected in {"answer", "redirect"},
        answered=actual in {"answer", "redirect"},
        passed=expected == actual,
        triggers=[],
        citations=["Lecture (lec01)"] if actual in {"answer", "redirect"} else [],
        latency_ms=1.0,
        answer="Fixture answer.",
        expected_behavior=expected,
        actual_behavior=actual,
        family_id=family_id,
    )


def test_bootstrap_reports_deterministic_row_and_family_intervals() -> None:
    results = [
        _result("a-1", "a", "answer", "answer"),
        _result("a-2", "a", "answer", "abstain"),
        _result("b-1", "b", "block", "block"),
        _result("b-2", "b", "block", "answer"),
        _result("c-1", "c", "abstain", "abstain"),
        _result("c-2", "c", "abstain", "answer"),
        _result("d-1", "d", "redirect", "redirect"),
        _result("d-2", "d", "redirect", "block"),
    ]

    first = bootstrap_confidence_intervals(results, samples=200, seed=17)
    second = bootstrap_confidence_intervals(results, samples=200, seed=17)

    assert first == second
    assert first["confidence_level"] == 0.95
    assert first["row"]["sampling_units"] == 8
    assert first["family"]["sampling_units"] == 4
    assert first["row"]["metrics"]["behavior_accuracy"]["point"] == 0.5
    assert first["family"]["metrics"]["macro_behavior_f1"]["samples_used"] == 200
    for scope in ("row", "family"):
        interval = first[scope]["metrics"]["behavior_accuracy"]
        assert interval["lower"] <= interval["point"] <= interval["upper"]
