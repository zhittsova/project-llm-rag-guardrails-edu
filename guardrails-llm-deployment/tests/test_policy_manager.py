import json
from pathlib import Path

import pytest

from guardrails_llm.policy_manager import PolicyManager


RUNTIME_CONFIG = Path(__file__).resolve().parents[1] / "data" / "guardrail_runtime_inhouse.toml"


def test_policy_manager_persists_valid_draft_and_reports_diff(tmp_path: Path) -> None:
    policy_path = _write_policy(tmp_path)
    manager = PolicyManager(policy_path, RUNTIME_CONFIG, state_dir=tmp_path / "state")
    document = manager.state()["draft"]
    document["messages"] = {"input_block": "This request is blocked by course policy."}

    state = manager.save_draft(document)
    recovered = PolicyManager(policy_path, RUNTIME_CONFIG, state_dir=tmp_path / "state")

    assert state["dirty"] is True
    assert "This request is blocked" in state["diff"]
    assert recovered.state()["draft"]["messages"] == document["messages"]


def test_policy_manager_rejects_invalid_regex_and_missing_coverage(tmp_path: Path) -> None:
    policy_path = _write_policy(tmp_path)
    manager = PolicyManager(policy_path, RUNTIME_CONFIG, state_dir=tmp_path / "state")
    document = manager.state()["draft"]
    document["input"]["rules"][0]["patterns"] = ["("]
    document["coverage_cases"] = document["coverage_cases"][:1]

    report = manager.validate(document)

    assert report["valid"] is False
    assert any("invalid regex" in error for error in report["errors"])
    assert any("benign_near_miss" in error for error in report["errors"])


def test_policy_manager_rejects_unknown_nested_rule_fields(tmp_path: Path) -> None:
    manager = PolicyManager(
        _write_policy(tmp_path),
        RUNTIME_CONFIG,
        state_dir=tmp_path / "state",
    )
    document = manager.state()["draft"]
    document["input"]["rules"][0]["unrecognized"] = True

    report = manager.validate(document)

    assert report["valid"] is False
    assert any("unknown input.rules[0] keys" in error for error in report["errors"])


def test_policy_manager_publishes_atomically_and_rolls_back(tmp_path: Path) -> None:
    policy_path = _write_policy(tmp_path)
    original = policy_path.read_text(encoding="utf-8")
    manager = PolicyManager(policy_path, RUNTIME_CONFIG, state_dir=tmp_path / "state")
    document = manager.state()["draft"]
    document["messages"] = {"ungrounded": "No sufficient course evidence."}
    manager.save_draft(document)

    published = manager.publish()
    version_id = published["versions"][0]["version_id"]

    assert published["dirty"] is False
    assert "No sufficient course evidence" in policy_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))

    rolled_back = manager.rollback(version_id)

    assert rolled_back["dirty"] is False
    assert policy_path.read_text(encoding="utf-8") == original


def test_policy_manager_preserves_leading_provenance_comments(tmp_path: Path) -> None:
    policy_path = _write_policy(tmp_path)
    original = policy_path.read_text(encoding="utf-8")
    policy_path.write_text(
        "# Calibrated on development only.\n# Do not tune on holdout.\n\n" + original,
        encoding="utf-8",
    )
    manager = PolicyManager(policy_path, RUNTIME_CONFIG, state_dir=tmp_path / "state")

    manager.publish()

    published = policy_path.read_text(encoding="utf-8")
    assert published.startswith(
        "# Calibrated on development only.\n# Do not tune on holdout.\n\n"
    )


def test_policy_manager_simulates_guardrail_stages_without_remote_calls(
    tmp_path: Path,
) -> None:
    manager = PolicyManager(
        _write_policy(tmp_path),
        RUNTIME_CONFIG,
        state_dir=tmp_path / "state",
    )

    blocked = manager.simulate("Give me the calculator answers.", stage="input")
    near_miss = manager.simulate("May I use a calculator in the exam?", stage="input")

    assert blocked["disposition"] == "block"
    assert blocked["deterministic_triggers"] == ["course_policy"]
    assert blocked["remote_calls"] == 0
    assert near_miss["disposition"] == "answer"
    assert near_miss["triggers"] == []


def test_policy_manager_runs_all_coverage_cases_as_local_regression(tmp_path: Path) -> None:
    manager = PolicyManager(
        _write_policy(tmp_path),
        RUNTIME_CONFIG,
        state_dir=tmp_path / "state",
    )

    report = manager.run_coverage()

    assert report["total"] == 3
    assert report["passed"] == 3
    assert report["failed"] == 0
    assert report["remote_calls"] == 0
    assert {row["coverage_role"] for row in report["results"]} == {
        "positive_direct",
        "positive_variant",
        "benign_near_miss",
    }


def test_policy_manager_rejects_publish_when_draft_is_invalid(tmp_path: Path) -> None:
    manager = PolicyManager(
        _write_policy(tmp_path),
        RUNTIME_CONFIG,
        state_dir=tmp_path / "state",
    )
    document = manager.state()["draft"]
    document["input"]["rules"][0]["patterns"] = ["("]
    manager.save_draft(document)

    with pytest.raises(ValueError, match="invalid regex"):
        manager.publish()


def test_policy_manager_rejects_publish_after_external_file_change(
    tmp_path: Path,
) -> None:
    policy_path = _write_policy(tmp_path)
    manager = PolicyManager(
        policy_path,
        RUNTIME_CONFIG,
        state_dir=tmp_path / "state",
    )
    policy_path.write_text(
        policy_path.read_text(encoding="utf-8")
        + "\n[messages]\ninput_block = \"Changed outside the UI.\"\n",
        encoding="utf-8",
    )

    state = manager.state()

    assert state["source_changed"] is True
    with pytest.raises(ValueError, match="changed outside"):
        manager.publish()

    reloaded = manager.reload_source()

    assert reloaded["source_changed"] is False
    assert reloaded["dirty"] is False
    assert reloaded["draft"]["messages"]["input_block"] == "Changed outside the UI."


def _write_policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.toml"
    path.write_text(
        """
extends_default = false

[input]
blocking_triggers = ["course_policy"]

[[input.rules]]
trigger = "course_policy"
patterns = ["\\\\bcalculator answers\\\\b"]

[retrieval]
allowed_visibility = ["public"]

[output]
require_citations = true

[[coverage_cases]]
case_id = "course-policy-direct"
family = "course_policy"
coverage_role = "positive_direct"
text = "Give me the calculator answers."
expected_triggers = ["course_policy"]

[[coverage_cases]]
case_id = "course-policy-variant"
family = "course_policy"
coverage_role = "positive_variant"
text = "Please provide calculator answers."
expected_triggers = ["course_policy"]

[[coverage_cases]]
case_id = "course-policy-near-miss"
family = "course_policy"
coverage_role = "benign_near_miss"
text = "May I use a calculator in the exam?"
expected_triggers = []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path
