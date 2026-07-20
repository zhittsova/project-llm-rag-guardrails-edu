from __future__ import annotations

import json
from pathlib import Path

from guardrails_llm.judge_study_audit import audit_judge_study


def test_audit_reports_duplicate_and_language_quality_failures(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    item = {
        "schema_version": 1,
        "item_id": "judge_calibration-one",
        "judge_split": "judge_calibration",
        "question": "Eine Person fragt bei die Schleife nach eine Antwort.",
        "language": "de",
        "expected_behavior": "answer",
        "actual_behavior": "answer",
        "answer": "Example",
        "retrieved_evidence": [],
        "supporting_chunks": [],
    }
    duplicate = {**item, "item_id": "judge_calibration-two"}
    _write_jsonl(study_dir / "judge_calibration_items.jsonl", [item, duplicate])
    _write_jsonl(study_dir / "judge_validation_items.jsonl", [])
    _write_jsonl(
        study_dir / "judge_study_mapping.jsonl",
        [
            {"item_id": item["item_id"], "scenario": "baseline"},
            {"item_id": duplicate["item_id"], "scenario": "baseline"},
        ],
    )

    report = audit_judge_study(study_dir)

    assert report["items"] == 2
    assert report["exact_duplicate_items"] == 1
    assert report["german_grammar_issue_items"] == 2
    assert report["complete_hybrid_items"] == 0
    assert report["quality_gates_passed"] is False


def test_audit_accepts_distinct_grounded_model_outputs(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    items = []
    mappings = []
    for split in ("judge_calibration", "judge_validation"):
        split_items = []
        for index in range(4):
            item_id = f"{split}-{index}"
            item = {
                "schema_version": 1,
                "item_id": item_id,
                "judge_split": split,
                "question": f"Question {split} {index}",
                "language": "en",
                "expected_behavior": "answer",
                "actual_behavior": "answer",
                "answer": f"Answer {index}",
                "retrieved_evidence": [{"chunk_id": f"chunk-{index}"}],
                "supporting_chunks": [f"chunk-{index}"],
            }
            split_items.append(item)
            items.append(item)
            mappings.append(
                {"item_id": item_id, "scenario": "complete_inhouse_hybrid"}
            )
        _write_jsonl(study_dir / f"{split}_items.jsonl", split_items)
    _write_jsonl(study_dir / "judge_study_mapping.jsonl", mappings)

    report = audit_judge_study(study_dir, minimum_items_per_split=4)

    assert report["exact_duplicate_items"] == 0
    assert report["complete_hybrid_items"] == len(items)
    assert report["supporting_chunk_items"] == len(items)
    assert report["quality_gates_passed"] is True


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
