from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardrails_llm.review_store import ReviewStore


DIMENSIONS = (
    "grounded",
    "privacy_safe",
    "injection_safe",
    "integrity_safe",
    "refusal_appropriate",
)


def test_store_recovers_autosaved_draft_after_restart(tmp_path: Path) -> None:
    study_dir = _study_dir(tmp_path)
    store = ReviewStore(study_dir, "reviewer_a", section_size=1)
    item_id = store.sections()[0]["item_ids"][0]

    store.save_draft(
        item_id,
        {
            "annotator_id": "alice",
            "grounded": True,
            "rationale": "The cited evidence supports the response.",
        },
    )

    recovered = ReviewStore(study_dir, "reviewer_a", section_size=1)
    draft = recovered.draft(item_id)
    assert draft["annotator_id"] == "alice"
    assert draft["grounded"] is True
    assert draft["privacy_safe"] is None
    assert draft["rationale"] == "The cited evidence supports the response."


def test_store_keeps_reviewer_drafts_isolated(tmp_path: Path) -> None:
    study_dir = _study_dir(tmp_path)
    reviewer_a = ReviewStore(study_dir, "reviewer_a", section_size=1)
    reviewer_b = ReviewStore(study_dir, "reviewer_b", section_size=1)
    item_id = reviewer_a.sections()[0]["item_ids"][0]

    reviewer_a.save_draft(item_id, {"annotator_id": "alice", "grounded": True})

    assert reviewer_a.draft(item_id)["grounded"] is True
    assert reviewer_b.draft(item_id)["grounded"] is None
    assert reviewer_b.draft(item_id)["annotator_id"] == ""


def test_complete_section_flushes_annotations_atomically(tmp_path: Path) -> None:
    study_dir = _study_dir(tmp_path)
    store = ReviewStore(study_dir, "reviewer_a", section_size=1)
    section = store.sections()[0]
    item_id = section["item_ids"][0]
    annotation = {
        "annotator_id": "alice",
        **{dimension: True for dimension in DIMENSIONS},
        "rationale": "Reviewed against the evidence and expected behavior.",
    }

    result = store.save_draft(item_id, annotation)

    assert result["section_flushed"] is True
    output = _read_jsonl(study_dir / "judge_calibration_reviewer_a.jsonl")
    saved = next(row for row in output if row["item_id"] == item_id)
    assert saved == {"item_id": item_id, **annotation}
    assert not list(study_dir.glob("*.tmp"))


def test_incomplete_section_is_not_flushed(tmp_path: Path) -> None:
    study_dir = _study_dir(tmp_path)
    store = ReviewStore(study_dir, "reviewer_a", section_size=1)
    section = store.sections()[0]
    item_id = section["item_ids"][0]

    result = store.save_draft(item_id, {"annotator_id": "alice", "grounded": True})

    assert result["section_flushed"] is False
    output = _read_jsonl(study_dir / "judge_calibration_reviewer_a.jsonl")
    saved = next(row for row in output if row["item_id"] == item_id)
    assert saved["grounded"] is None


def test_flagged_issue_is_exported_without_forcing_labels(tmp_path: Path) -> None:
    study_dir = _study_dir(tmp_path)
    store = ReviewStore(study_dir, "reviewer_a", section_size=1)
    section = store.sections()[0]
    item_id = section["item_ids"][0]

    result = store.save_draft(
        item_id,
        {
            "annotator_id": "alice",
            "issue_flag": True,
            "issue_note": "The German prompt is not grammatical enough to judge.",
        },
    )

    assert result["section_flushed"] is True
    issue_rows = _read_jsonl(study_dir / "judge_reviewer_a_issues.jsonl")
    assert issue_rows == [
        {
            "item_id": item_id,
            "reviewer": "reviewer_a",
            "annotator_id": "alice",
            "issue_note": "The German prompt is not grammatical enough to judge.",
        }
    ]
    annotation_rows = _read_jsonl(
        study_dir / "judge_calibration_reviewer_a.jsonl"
    )
    saved = next(row for row in annotation_rows if row["item_id"] == item_id)
    assert all(saved[dimension] is None for dimension in DIMENSIONS)


def test_store_rejects_unknown_reviewer_and_item(tmp_path: Path) -> None:
    study_dir = _study_dir(tmp_path)

    with pytest.raises(ValueError, match="reviewer_a or reviewer_b"):
        ReviewStore(study_dir, "reviewer_c")

    store = ReviewStore(study_dir, "reviewer_a")
    with pytest.raises(KeyError, match="unknown study item"):
        store.save_draft("not-an-item", {"grounded": True})


def _study_dir(tmp_path: Path) -> Path:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    for split in ("judge_calibration", "judge_validation"):
        items = [
            {
                "schema_version": 1,
                "item_id": f"{split}-one",
                "judge_split": split,
                "question": f"Question for {split}",
                "language": "en",
                "expected_behavior": "answer",
                "actual_behavior": "answer",
                "answer": "A grounded answer.",
                "retrieved_evidence": [{"chunk_id": "chunk-1", "text": "Evidence"}],
                "supporting_chunks": ["chunk-1"],
            },
            {
                "schema_version": 1,
                "item_id": f"{split}-two",
                "judge_split": split,
                "question": f"Second question for {split}",
                "language": "en",
                "expected_behavior": "block",
                "actual_behavior": "block",
                "answer": "Request blocked.",
                "retrieved_evidence": [],
                "supporting_chunks": [],
            },
        ]
        _write_jsonl(study_dir / f"{split}_items.jsonl", items)
        for reviewer in ("reviewer_a", "reviewer_b"):
            _write_jsonl(
                study_dir / f"{split}_{reviewer}.jsonl",
                [
                    {
                        "item_id": item["item_id"],
                        "annotator_id": "",
                        **{dimension: None for dimension in DIMENSIONS},
                        "rationale": "",
                    }
                    for item in items
                ],
            )
    return study_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
