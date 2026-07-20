from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from .judge_study import (
    JUDGE_SPLITS,
    reconcile_human_annotations,
    validate_annotation_file,
)
from .model_calibration import JUDGE_DIMENSIONS
from .review_recommendations import RecommendationStore


class ReconciliationStore:
    def __init__(self, study_dir: Path, *, section_size: int = 10) -> None:
        if section_size < 1:
            raise ValueError("section_size must be positive")
        self.study_dir = study_dir.resolve()
        self.section_size = section_size
        self._items = {
            str(item["item_id"]): item
            for split in JUDGE_SPLITS
            for item in _load_jsonl(
                self.study_dir / f"{split}_items.jsonl"
            )
        }
        expected_ids = set(self._items)
        self._reviewers = {}
        for reviewer in ("reviewer_a", "reviewer_b"):
            rows = []
            for split in JUDGE_SPLITS:
                path = self.study_dir / f"{split}_{reviewer}.jsonl"
                try:
                    annotations, summary = validate_annotation_file(
                        path,
                        expected_item_ids={
                            item_id
                            for item_id in expected_ids
                            if item_id.startswith(f"{split}-")
                        },
                        complete=True,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{reviewer} review is incomplete: {exc}"
                    ) from exc
                if not summary["complete"]:
                    raise ValueError(f"{reviewer} review is incomplete")
                rows.extend(
                    {
                        "item_id": annotation.item_id,
                        "annotator_id": annotation.annotator_id,
                        **{
                            dimension: getattr(annotation, dimension)
                            for dimension in JUDGE_DIMENSIONS
                        },
                        "rationale": annotation.rationale,
                    }
                    for annotation in annotations
                )
            self._reviewers[reviewer] = {
                str(row["item_id"]): row for row in rows
            }
        self._recommendations = RecommendationStore(self.study_dir)
        disagreements_path = self.study_dir / "judge_disagreements.jsonl"
        previous = (
            {
                str(row["item_id"]): row
                for row in _load_jsonl(disagreements_path)
            }
            if disagreements_path.exists()
            else {}
        )
        reconcile_human_annotations(
            items_paths=[
                self.study_dir / f"{split}_items.jsonl"
                for split in JUDGE_SPLITS
            ],
            reviewer_a_paths=[
                self.study_dir / f"{split}_reviewer_a.jsonl"
                for split in JUDGE_SPLITS
            ],
            reviewer_b_paths=[
                self.study_dir / f"{split}_reviewer_b.jsonl"
                for split in JUDGE_SPLITS
            ],
            disagreements_output=disagreements_path,
            report_output=self.study_dir / "judge_human_agreement.json",
        )
        self._disagreements = {
            str(row["item_id"]): row
            for row in _load_jsonl(disagreements_path)
        }
        for item_id, old_row in previous.items():
            if item_id not in self._disagreements:
                continue
            self._disagreements[item_id].update(
                {
                    key: old_row[key]
                    for key in ("adjudicator_id", *JUDGE_DIMENSIONS, "rationale")
                    if key in old_row
                }
            )
        _atomic_write_jsonl(
            disagreements_path,
            [
                self._disagreements[key]
                for key in sorted(self._disagreements)
            ],
        )
        self._sections = self._build_sections()

    def sections(self) -> list[dict[str, object]]:
        result = []
        for section in self._sections:
            disagreement_ids = [
                item_id
                for item_id in section["item_ids"]
                if item_id in self._disagreements
            ]
            completed = sum(
                _adjudication_complete(self._disagreements[item_id])
                for item_id in disagreement_ids
            )
            result.append(
                {
                    **section,
                    "disagreements": len(disagreement_ids),
                    "completed": completed,
                    "remaining": len(disagreement_ids) - completed,
                }
            )
        return result

    def item(self, item_id: str) -> dict[str, object]:
        try:
            item = self._items[item_id]
        except KeyError as exc:
            raise KeyError(f"unknown reconciliation item: {item_id}") from exc
        return {
            **item,
            "reviewer_a": self._reviewers["reviewer_a"][item_id],
            "reviewer_b": self._reviewers["reviewer_b"][item_id],
            "recommendation": self._recommendations.get(item_id),
            "requires_adjudication": item_id in self._disagreements,
            "adjudication": self._disagreements.get(item_id),
        }

    def section(self, section_id: str) -> dict[str, object]:
        section = next(
            (
                section
                for section in self.sections()
                if section["section_id"] == section_id
            ),
            None,
        )
        if section is None:
            raise KeyError(f"unknown reconciliation section: {section_id}")
        return {
            "section": section,
            "items": [
                self.item(str(item_id))
                for item_id in section["item_ids"]
            ],
        }

    def progress(self) -> dict[str, int]:
        total = len(self._disagreements)
        completed = sum(
            _adjudication_complete(row)
            for row in self._disagreements.values()
        )
        return {
            "total": total,
            "completed": completed,
            "remaining": total - completed,
        }

    def save_adjudication(
        self,
        item_id: str,
        changes: dict[str, object],
    ) -> dict[str, object]:
        if item_id not in self._disagreements:
            raise ValueError(
                "adjudication is allowed only for human-review disagreements"
            )
        allowed = {"adjudicator_id", "rationale", *JUDGE_DIMENSIONS}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(
                f"unknown adjudication fields: {', '.join(sorted(unknown))}"
            )
        for dimension in JUDGE_DIMENSIONS:
            if dimension in changes and changes[dimension] not in (
                True,
                False,
                None,
            ):
                raise ValueError(
                    f"{dimension} must be true, false, or null"
                )
        for field in ("adjudicator_id", "rationale"):
            if field in changes and not isinstance(changes[field], str):
                raise ValueError(f"{field} must be a string")
        self._disagreements[item_id].update(changes)
        _atomic_write_jsonl(
            self.study_dir / "judge_disagreements.jsonl",
            [
                self._disagreements[key]
                for key in sorted(self._disagreements)
            ],
        )
        return {
            "item": self.item(item_id),
            "progress": self.progress(),
        }

    def _build_sections(self) -> list[dict[str, object]]:
        groups: OrderedDict[str, list[str]] = OrderedDict()
        for item_id, item in self._items.items():
            groups.setdefault(str(item["judge_split"]), []).append(item_id)
        sections = []
        for split, item_ids in groups.items():
            for offset in range(0, len(item_ids), self.section_size):
                number = offset // self.section_size + 1
                sections.append(
                    {
                        "section_id": f"{split}-{number:02d}",
                        "split": split,
                        "title": (
                            f"{'Calibration' if split == 'judge_calibration' else 'Validation'} "
                            f"{number}"
                        ),
                        "item_ids": item_ids[
                            offset : offset + self.section_size
                        ],
                    }
                )
        return sections


def _adjudication_complete(row: dict[str, object]) -> bool:
    return (
        bool(str(row.get("adjudicator_id", "")).strip())
        and bool(str(row.get("rationale", "")).strip())
        and all(
            isinstance(row.get(dimension), bool)
            for dimension in JUDGE_DIMENSIONS
        )
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
