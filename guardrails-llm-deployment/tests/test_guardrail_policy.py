from pathlib import Path

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
