from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .judge_study import JUDGE_SPLITS


GERMAN_GRAMMAR_ISSUES = (
    re.compile(r"\bmit die\b", re.IGNORECASE),
    re.compile(r"\bbei die\b", re.IGNORECASE),
    re.compile(r"\bnach eine\b", re.IGNORECASE),
    re.compile(r"\bsollst du für\b", re.IGNORECASE),
    re.compile(r"\b(?:zeige|liste|rufe|gib)\b", re.IGNORECASE),
)


def audit_judge_study(
    study_dir: Path,
    *,
    minimum_items_per_split: int = 200,
) -> dict[str, object]:
    items_by_split = {
        split: _load_jsonl(study_dir / f"{split}_items.jsonl")
        for split in JUDGE_SPLITS
    }
    items = [item for split_items in items_by_split.values() for item in split_items]
    mappings = _load_jsonl(study_dir / "judge_study_mapping.jsonl")
    scenarios = Counter(str(mapping.get("scenario", "")) for mapping in mappings)

    signatures = Counter(_visible_signature(item) for item in items)
    exact_duplicate_items = sum(count - 1 for count in signatures.values() if count > 1)
    german_grammar_issue_ids = [
        str(item.get("item_id", ""))
        for item in items
        if item.get("language") == "de"
        and any(pattern.search(str(item.get("question", ""))) for pattern in GERMAN_GRAMMAR_ISSUES)
    ]
    supporting_chunk_items = sum(bool(item.get("supporting_chunks")) for item in items)
    split_quality = {}
    for split, split_items in items_by_split.items():
        unique_questions = len({str(item.get("question", "")) for item in split_items})
        split_quality[split] = {
            "items": len(split_items),
            "unique_questions": unique_questions,
            "unique_question_ratio": round(unique_questions / len(split_items), 3)
            if split_items
            else 0.0,
        }

    gates = {
        "required_split_sizes": all(
            metrics["items"] >= minimum_items_per_split
            for metrics in split_quality.values()
        ),
        "no_exact_duplicate_tasks": exact_duplicate_items == 0,
        "unique_question_ratio": all(
            metrics["unique_question_ratio"] >= 0.8
            for metrics in split_quality.values()
        ),
        "german_template_grammar": not german_grammar_issue_ids,
        "complete_hybrid_coverage": scenarios["complete_inhouse_hybrid"] > 0,
        "supporting_chunk_coverage": supporting_chunk_items > 0,
    }
    return {
        "schema_version": 1,
        "items": len(items),
        "split_quality": split_quality,
        "exact_duplicate_items": exact_duplicate_items,
        "german_grammar_issue_items": len(german_grammar_issue_ids),
        "german_grammar_issue_item_ids": german_grammar_issue_ids,
        "scenario_counts": dict(sorted(scenarios.items())),
        "complete_hybrid_items": scenarios["complete_inhouse_hybrid"],
        "supporting_chunk_items": supporting_chunk_items,
        "quality_gates": gates,
        "quality_gates_passed": all(gates.values()),
    }


def _visible_signature(item: dict[str, object]) -> str:
    return json.dumps(
        {key: value for key, value in item.items() if key != "item_id"},
        sort_keys=True,
        ensure_ascii=False,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
