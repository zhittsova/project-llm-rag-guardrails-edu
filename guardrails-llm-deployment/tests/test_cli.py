from __future__ import annotations

import json
import sys
from pathlib import Path

from guardrails_llm.cli import main


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
