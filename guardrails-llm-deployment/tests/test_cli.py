from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from guardrails_llm.cli import main
from guardrails_llm.model_config import RemoteModelCallError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "course_docs.jsonl"
CASES = ROOT / "data" / "eval_cases.jsonl"


def test_compare_guardrails_writes_json_artifact(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "compare-guardrails",
            "--corpus",
            str(DATA),
            "--cases",
            str(CASES),
            "--limit-cases",
            "2",
            "--output-json",
            str(output),
        ],
    )

    main()

    data = json.loads(output.read_text(encoding="utf-8"))
    assert "baseline" in data
    assert "default_guardrails" in data
    assert "hybrid_policy_guardrails" in data


def test_build_index_reports_remote_model_error_without_traceback(monkeypatch, capsys) -> None:
    def fail_build(*args, **kwargs):
        raise RemoteModelCallError("OpenAI embedding request failed: AuthenticationError")

    monkeypatch.setattr("guardrails_llm.cli.build_vector_index", fail_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "build-index",
            "--corpus",
            str(DATA),
            "--embedding-provider",
            "openai",
            "--allow-remote-models",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "OpenAI embedding request failed: AuthenticationError" in captured.err
    assert "Traceback" not in captured.err


def test_remote_guard_embeddings_require_explicit_allowance(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "compare-guardrails",
            "--corpus",
            str(DATA),
            "--cases",
            str(CASES),
            "--guard-embedding-provider",
            "openai",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Remote model calls are disabled" in captured.err
    assert "Traceback" not in captured.err
