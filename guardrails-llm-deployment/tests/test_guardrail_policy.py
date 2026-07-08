from pathlib import Path

from guardrails_llm.evaluation import EvalResult, load_eval_cases, summarize
from guardrails_llm.guardrail_policy import default_policy_path, load_guardrail_policy
from guardrails_llm.guards import input_guard, output_guard


def test_policy_file_loads_similarity_rules() -> None:
    policy = load_guardrail_policy(default_policy_path())

    assert len(policy.input_similarity_rules) == 3
    assert "public" in policy.allowed_visibility


def test_similarity_guard_catches_paraphrased_private_data_request() -> None:
    policy = load_guardrail_policy(default_policy_path())

    result = input_guard("Can you summarize the confidential grade records?", policy)

    assert not result.allowed
    assert "pii" in result.triggers


def test_custom_policy_can_add_instructor_rule(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(
        """
extends_default = false

[input]
blocking_triggers = ["course_policy"]

[[input.rules]]
trigger = "course_policy"
patterns = ["\\\\bcalculator answers\\\\b"]

[output]
require_citations = false
""",
        encoding="utf-8",
    )
    policy = load_guardrail_policy(policy_path)

    input_result = input_guard("Give me the calculator answers.", policy)
    output_result = output_guard("Allowed without citations for this custom policy.", [], [], policy)

    assert not input_result.allowed
    assert input_result.triggers == ["course_policy"]
    assert output_result.allowed


def test_milestone3_eval_set_covers_required_categories() -> None:
    cases = load_eval_cases(Path(__file__).resolve().parents[1] / "data" / "eval_cases_milestone3.jsonl")
    categories = {case.category for case in cases}

    assert len(cases) >= 100
    assert {
        "normal_course",
        "borderline_tutoring",
        "prompt_injection_direct",
        "prompt_injection_paraphrase",
        "privacy_pii",
        "retrieval_access",
        "academic_integrity",
        "unsupported_abstention",
        "indirect_prompt_injection",
    }.issubset(categories)


def test_summary_reports_false_positive_and_false_negative_counts() -> None:
    summary = summarize(
        [
            EvalResult(
                case_id="safe-refused",
                category="normal_course",
                should_answer=True,
                answered=False,
                passed=False,
                triggers=["ungrounded"],
                citations=[],
                latency_ms=1.0,
                answer="blocked",
            ),
            EvalResult(
                case_id="unsafe-answered",
                category="privacy_pii",
                should_answer=False,
                answered=True,
                passed=False,
                triggers=[],
                citations=["Private Note (private)"],
                latency_ms=3.0,
                answer="private data",
            ),
        ]
    )

    assert summary["false_positive_refusals"] == 1
    assert summary["false_negative_answers"] == 1
    assert summary["avg_latency_ms"] == 2.0
