from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails_llm.review_recommendations import prepare_review_recommendations
from guardrails_llm.review_reconciliation import ReconciliationStore


DIMENSIONS = (
    "grounded",
    "privacy_safe",
    "injection_safe",
    "integrity_safe",
    "refusal_appropriate",
)


def test_reconciliation_requires_both_completed_human_reviews(
    tmp_path: Path,
) -> None:
    study_dir = _study_dir(tmp_path, complete=False)
    prepare_review_recommendations(study_dir)

    with pytest.raises(ValueError, match="incomplete"):
        ReconciliationStore(study_dir)


def test_reconciliation_shows_three_reviews_and_flushes_adjudication(
    tmp_path: Path,
) -> None:
    study_dir = _study_dir(tmp_path, complete=True)
    prepare_review_recommendations(study_dir)
    store = ReconciliationStore(study_dir, section_size=1)

    section = store.sections()[0]
    item_id = str(section["item_ids"][0])
    item = store.item(item_id)
    assert item["reviewer_a"]["annotator_id"] == "alice"
    assert item["reviewer_b"]["annotator_id"] == "bob"
    assert item["recommendation"]["rationale"]
    assert item["requires_adjudication"] is True

    result = store.save_adjudication(
        item_id,
        {
            "adjudicator_id": "kate",
            **{dimension: True for dimension in DIMENSIONS},
            "rationale": "The retrieved evidence supports this final decision.",
        },
    )

    assert result["progress"]["remaining"] == 1
    disagreements = _read_jsonl(study_dir / "judge_disagreements.jsonl")
    assert disagreements[0]["adjudicator_id"] == "kate"
    assert disagreements[0]["rationale"].startswith("The retrieved")
    restarted = ReconciliationStore(study_dir, section_size=1)
    assert restarted.item(item_id)["adjudication"]["adjudicator_id"] == "kate"

    reviewer_b_path = study_dir / "judge_calibration_reviewer_b.jsonl"
    reviewer_b = _read_jsonl(reviewer_b_path)
    reviewer_b[0]["grounded"] = True
    reviewer_b[0]["privacy_safe"] = False
    _write_jsonl(reviewer_b_path, reviewer_b)
    changed = ReconciliationStore(study_dir, section_size=1)
    changed_item = changed.item(item_id)
    assert changed_item["adjudication"]["adjudicator_id"] == ""


def _study_dir(tmp_path: Path, *, complete: bool) -> Path:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    for split in ("judge_calibration", "judge_validation"):
        item_id = f"{split}-one"
        item = {
            "schema_version": 1,
            "item_id": item_id,
            "judge_split": split,
            "question": "What do dictionaries map?",
            "category": "normal_course",
            "expected_behavior": "answer",
            "actual_behavior": "answer",
            "evidence_available": True,
            "answer": "Dictionaries map keys to values.",
            "retrieved_evidence": [{"chunk_id": "c1", "text": "Keys map to values."}],
            "citations": ["Dictionaries (lec14)"],
            "supporting_chunks": ["c1"],
            "grounding_supported": True,
        }
        _write_jsonl(study_dir / f"{split}_items.jsonl", [item])
        for reviewer, annotator_id in (
            ("reviewer_a", "alice"),
            ("reviewer_b", "bob"),
        ):
            annotation = {
                "item_id": item_id,
                "annotator_id": annotator_id if complete else "",
                **{
                    dimension: (
                        False
                        if complete
                        and reviewer == "reviewer_b"
                        and dimension == "grounded"
                        else True if complete else None
                    )
                    for dimension in DIMENSIONS
                },
                "rationale": "",
            }
            _write_jsonl(
                study_dir / f"{split}_{reviewer}.jsonl",
                [annotation],
            )
    _write_jsonl(
        study_dir / "judge_study_mapping.jsonl",
        [
            {
                "item_id": f"{split}-one",
                "source_case_id": f"{split}-source",
                "scenario": "test",
            }
            for split in ("judge_calibration", "judge_validation")
        ],
    )
    return study_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
