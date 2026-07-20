from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .model_calibration import JUDGE_DIMENSIONS
from .review_recommendations import RecommendationStore


REVIEWERS = ("reviewer_a", "reviewer_b")
EDITABLE_FIELDS = (
    "annotator_id",
    *JUDGE_DIMENSIONS,
    "rationale",
    "issue_flag",
    "issue_note",
)


class ReviewStore:
    def __init__(
        self,
        study_dir: Path,
        reviewer: str,
        *,
        section_size: int = 10,
    ) -> None:
        if reviewer not in REVIEWERS:
            raise ValueError("reviewer must be reviewer_a or reviewer_b")
        if section_size < 1:
            raise ValueError("section_size must be positive")
        self.study_dir = study_dir.resolve()
        self.reviewer = reviewer
        self.section_size = section_size
        self.database_path = self.study_dir / ".judge_review.sqlite3"
        self._items, self._sections, self._section_by_item = self._load_study()
        self._recommendations = self._load_recommendations()
        self._initialize_database()

    def sections(self) -> list[dict[str, object]]:
        states = self._draft_states()
        result = []
        for section in self._sections:
            item_ids = section["item_ids"]
            complete = sum(_is_complete(states[item_id]) for item_id in item_ids)
            issues = sum(_is_issue(states[item_id]) for item_id in item_ids)
            resolved = complete + issues
            result.append(
                {
                    **section,
                    "completed": complete,
                    "issues": issues,
                    "remaining": len(item_ids) - resolved,
                    "ready": resolved == len(item_ids),
                    "flushed": self._section_flushed(str(section["section_id"])),
                }
            )
        return result

    def items(self) -> list[dict[str, object]]:
        return list(self._items.values())

    def draft(self, item_id: str) -> dict[str, object]:
        self._require_item(item_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT annotator_id, grounded, privacy_safe, injection_safe,
                       integrity_safe, refusal_appropriate, rationale,
                       issue_flag, issue_note, updated_at
                FROM review_drafts
                WHERE reviewer = ? AND item_id = ?
                """,
                (self.reviewer, item_id),
            ).fetchone()
        assert row is not None
        return _row_to_draft(item_id, row)

    def save_draft(
        self,
        item_id: str,
        changes: dict[str, object],
    ) -> dict[str, object]:
        self._require_item(item_id)
        normalized = _validate_changes(changes)
        if not normalized:
            raise ValueError("draft changes must contain at least one editable field")
        assignments = ", ".join(f"{field} = ?" for field in normalized)
        values = [_sqlite_value(normalized[field]) for field in normalized]
        updated_at = _now()
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE review_drafts
                SET {assignments}, updated_at = ?
                WHERE reviewer = ? AND item_id = ?
                """,
                (*values, updated_at, self.reviewer, item_id),
            )
            connection.commit()
        section_id = self._section_by_item[item_id]
        section_flushed = False
        if self._section_ready(section_id):
            self.flush_section(section_id)
            section_flushed = True
        return {
            "draft": self.draft(item_id),
            "section_id": section_id,
            "section_flushed": section_flushed,
            "progress": self.progress(),
        }

    def set_annotator_id(self, annotator_id: str) -> dict[str, object]:
        if not isinstance(annotator_id, str) or not annotator_id.strip():
            raise ValueError("annotator_id must be a non-empty string")
        normalized = annotator_id.strip()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE review_drafts
                SET annotator_id = ?, updated_at = ?
                WHERE reviewer = ?
                """,
                (normalized, _now(), self.reviewer),
            )
            connection.commit()
        for section in self._sections:
            section_id = str(section["section_id"])
            if self._section_ready(section_id):
                self.flush_section(section_id)
        return {"annotator_id": normalized, "progress": self.progress()}

    def reveal_recommendation(self, item_id: str) -> dict[str, object]:
        self._require_item(item_id)
        if self._recommendations is None:
            raise FileNotFoundError("judge recommendation files are not available")
        self._record_assistance("recommendation_revealed", item_id=item_id)
        return self._recommendations.get(item_id)

    def apply_recommendation(self, item_id: str) -> dict[str, object]:
        self._require_item(item_id)
        if self._recommendations is None:
            raise FileNotFoundError("judge recommendation files are not available")
        recommendation = self._recommendations.get(item_id)
        result = self.save_draft(
            item_id,
            {
                dimension: recommendation[dimension]
                for dimension in JUDGE_DIMENSIONS
            },
        )
        self._record_assistance("recommendation_copied", item_id=item_id)
        return result

    def reveal_section_recommendations(
        self,
        section_id: str,
    ) -> dict[str, dict[str, object]]:
        section = next(
            (item for item in self._sections if item["section_id"] == section_id),
            None,
        )
        if section is None:
            raise KeyError(f"unknown review section: {section_id}")
        if self._recommendations is None:
            raise FileNotFoundError("judge recommendation files are not available")
        self._record_assistance(
            "section_recommendations_revealed",
            section_id=section_id,
        )
        return {
            str(item_id): self._recommendations.get(str(item_id))
            for item_id in section["item_ids"]
        }

    def reveal_all_recommendations(self) -> dict[str, dict[str, object]]:
        if self._recommendations is None:
            raise FileNotFoundError("judge recommendation files are not available")
        self._record_assistance("all_recommendations_revealed")
        return {
            str(item["item_id"]): item
            for item in self._recommendations.items()
        }

    def apply_section_recommendations(
        self,
        section_id: str,
    ) -> dict[str, object]:
        section = next(
            (item for item in self._sections if item["section_id"] == section_id),
            None,
        )
        if section is None:
            raise KeyError(f"unknown review section: {section_id}")
        if self._recommendations is None:
            raise FileNotFoundError("judge recommendation files are not available")
        for item_id in section["item_ids"]:
            recommendation = self._recommendations.get(str(item_id))
            self.save_draft(
                str(item_id),
                {
                    dimension: recommendation[dimension]
                    for dimension in JUDGE_DIMENSIONS
                },
            )
        self._record_assistance(
            "section_recommendations_copied",
            section_id=section_id,
        )
        return {
            "section_id": section_id,
            "copied": len(section["item_ids"]),
            "progress": self.progress(),
        }

    def assistance_events(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT reviewer, action, item_id, section_id, created_at
                FROM review_assistance_events
                WHERE reviewer = ?
                ORDER BY event_id
                """,
                (self.reviewer,),
            ).fetchall()
        return [dict(row) for row in rows]

    def progress(self) -> dict[str, int]:
        states = self._draft_states()
        complete = sum(_is_complete(state) for state in states.values())
        issues = sum(_is_issue(state) for state in states.values())
        return {
            "total": len(states),
            "completed": complete,
            "issues": issues,
            "remaining": len(states) - complete - issues,
        }

    def recommendations_available(self) -> bool:
        return self._recommendations is not None

    def flush_section(self, section_id: str) -> None:
        section = next(
            (item for item in self._sections if item["section_id"] == section_id),
            None,
        )
        if section is None:
            raise KeyError(f"unknown review section: {section_id}")
        if not self._section_ready(section_id):
            raise ValueError(f"section is incomplete: {section_id}")

        split = str(section["split"])
        item_ids = set(section["item_ids"])
        output_path = self.study_dir / f"{split}_{self.reviewer}.jsonl"
        output_rows = _load_jsonl(output_path)
        drafts = {item_id: self.draft(item_id) for item_id in item_ids}
        for row in output_rows:
            item_id = str(row.get("item_id", ""))
            draft = drafts.get(item_id)
            if draft is None or _is_issue(draft):
                continue
            row.update(_annotation_payload(draft))
        _atomic_write_jsonl(output_path, output_rows)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO section_exports(reviewer, section_id, flushed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(reviewer, section_id)
                DO UPDATE SET flushed_at = excluded.flushed_at
                """,
                (self.reviewer, section_id, _now()),
            )
            connection.commit()
        self._flush_issues()

    def _load_study(
        self,
    ) -> tuple[
        OrderedDict[str, dict[str, object]],
        list[dict[str, object]],
        dict[str, str],
    ]:
        items: OrderedDict[str, dict[str, object]] = OrderedDict()
        sections: list[dict[str, object]] = []
        section_by_item: dict[str, str] = {}
        for split in ("judge_calibration", "judge_validation"):
            split_items = _load_jsonl(self.study_dir / f"{split}_items.jsonl")
            question_groups: OrderedDict[str, list[str]] = OrderedDict()
            for item in split_items:
                item_id = str(item.get("item_id", ""))
                if not item_id or item_id in items:
                    raise ValueError("study items must have unique non-empty item_id values")
                items[item_id] = item
                question_groups.setdefault(str(item.get("question", "")), []).append(item_id)
            groups = list(question_groups.items())
            for offset in range(0, len(groups), self.section_size):
                section_number = offset // self.section_size + 1
                section_id = f"{split}-{section_number:02d}"
                batch = groups[offset : offset + self.section_size]
                item_ids = [item_id for _question, ids in batch for item_id in ids]
                sections.append(
                    {
                        "section_id": section_id,
                        "split": split,
                        "title": (
                            f"{'Calibration' if split == 'judge_calibration' else 'Validation'} "
                            f"{section_number}"
                        ),
                        "question_count": len(batch),
                        "item_ids": item_ids,
                    }
                )
                section_by_item.update({item_id: section_id for item_id in item_ids})
        if not items:
            raise ValueError("study directory contains no judge items")
        return items, sections, section_by_item

    def _initialize_database(self) -> None:
        self.study_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_drafts (
                    reviewer TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    annotator_id TEXT NOT NULL DEFAULT '',
                    grounded INTEGER,
                    privacy_safe INTEGER,
                    injection_safe INTEGER,
                    integrity_safe INTEGER,
                    refusal_appropriate INTEGER,
                    rationale TEXT NOT NULL DEFAULT '',
                    issue_flag INTEGER NOT NULL DEFAULT 0,
                    issue_note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (reviewer, item_id)
                );
                CREATE TABLE IF NOT EXISTS section_exports (
                    reviewer TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    flushed_at TEXT NOT NULL,
                    PRIMARY KEY (reviewer, section_id)
                );
                CREATE TABLE IF NOT EXISTS review_assistance_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reviewer TEXT NOT NULL,
                    action TEXT NOT NULL,
                    item_id TEXT,
                    section_id TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            for split in ("judge_calibration", "judge_validation"):
                annotations = {
                    str(row["item_id"]): row
                    for row in _load_jsonl(
                        self.study_dir / f"{split}_{self.reviewer}.jsonl"
                    )
                }
                for item_id in (
                    key for key in self._items if key.startswith(f"{split}-")
                ):
                    annotation = annotations.get(item_id, {})
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO review_drafts(
                            reviewer, item_id, section_id, annotator_id,
                            grounded, privacy_safe, injection_safe, integrity_safe,
                            refusal_appropriate, rationale, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.reviewer,
                            item_id,
                            self._section_by_item[item_id],
                            str(annotation.get("annotator_id", "")),
                            _sqlite_value(annotation.get("grounded")),
                            _sqlite_value(annotation.get("privacy_safe")),
                            _sqlite_value(annotation.get("injection_safe")),
                            _sqlite_value(annotation.get("integrity_safe")),
                            _sqlite_value(annotation.get("refusal_appropriate")),
                            str(annotation.get("rationale", "")),
                            _now(),
                        ),
                    )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _load_recommendations(self) -> RecommendationStore | None:
        paths = [
            self.study_dir / f"{split}_recommendation.jsonl"
            for split in ("judge_calibration", "judge_validation")
        ]
        if not any(path.exists() for path in paths):
            return None
        if not all(path.exists() for path in paths):
            raise ValueError("both recommendation split files are required")
        return RecommendationStore(self.study_dir)

    def _record_assistance(
        self,
        action: str,
        *,
        item_id: str | None = None,
        section_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO review_assistance_events(
                    reviewer, action, item_id, section_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (self.reviewer, action, item_id, section_id, _now()),
            )
            connection.commit()
        _atomic_write_jsonl(
            self.study_dir
            / f"judge_{self.reviewer}_recommendation_assistance.jsonl",
            self.assistance_events(),
        )

    def _draft_states(self) -> dict[str, dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT item_id, annotator_id, grounded, privacy_safe,
                       injection_safe, integrity_safe, refusal_appropriate,
                       rationale, issue_flag, issue_note, updated_at
                FROM review_drafts
                WHERE reviewer = ?
                """,
                (self.reviewer,),
            ).fetchall()
        return {
            str(row["item_id"]): _row_to_draft(str(row["item_id"]), row)
            for row in rows
        }

    def _section_ready(self, section_id: str) -> bool:
        item_ids = next(
            section["item_ids"]
            for section in self._sections
            if section["section_id"] == section_id
        )
        return all(
            _is_complete(draft) or _is_issue(draft)
            for draft in (self.draft(item_id) for item_id in item_ids)
        )

    def _section_flushed(self, section_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM section_exports
                WHERE reviewer = ? AND section_id = ?
                """,
                (self.reviewer, section_id),
            ).fetchone()
        return row is not None

    def _flush_issues(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.item_id, d.annotator_id, d.issue_note
                FROM review_drafts AS d
                JOIN section_exports AS e
                  ON e.reviewer = d.reviewer AND e.section_id = d.section_id
                WHERE d.reviewer = ? AND d.issue_flag = 1
                ORDER BY d.item_id
                """,
                (self.reviewer,),
            ).fetchall()
        issues = [
            {
                "item_id": str(row["item_id"]),
                "reviewer": self.reviewer,
                "annotator_id": str(row["annotator_id"]),
                "issue_note": str(row["issue_note"]),
            }
            for row in rows
        ]
        _atomic_write_jsonl(
            self.study_dir / f"judge_{self.reviewer}_issues.jsonl",
            issues,
        )

    def _require_item(self, item_id: str) -> None:
        if item_id not in self._items:
            raise KeyError(f"unknown study item: {item_id}")


def _validate_changes(changes: dict[str, object]) -> dict[str, object]:
    unknown = set(changes) - set(EDITABLE_FIELDS)
    if unknown:
        raise ValueError(f"unknown draft fields: {', '.join(sorted(unknown))}")
    normalized = dict(changes)
    for dimension in JUDGE_DIMENSIONS:
        if dimension in normalized and normalized[dimension] not in (True, False, None):
            raise ValueError(f"{dimension} must be true, false, or null")
    for field in ("annotator_id", "rationale", "issue_note"):
        if field in normalized and not isinstance(normalized[field], str):
            raise ValueError(f"{field} must be a string")
    if "issue_flag" in normalized and not isinstance(normalized["issue_flag"], bool):
        raise ValueError("issue_flag must be true or false")
    return normalized


def _row_to_draft(item_id: str, row: sqlite3.Row) -> dict[str, object]:
    return {
        "item_id": item_id,
        "annotator_id": str(row["annotator_id"]),
        **{dimension: _bool_or_none(row[dimension]) for dimension in JUDGE_DIMENSIONS},
        "rationale": str(row["rationale"]),
        "issue_flag": bool(row["issue_flag"]),
        "issue_note": str(row["issue_note"]),
        "updated_at": str(row["updated_at"]),
    }


def _annotation_payload(draft: dict[str, object]) -> dict[str, object]:
    return {
        "item_id": draft["item_id"],
        "annotator_id": draft["annotator_id"],
        **{dimension: draft[dimension] for dimension in JUDGE_DIMENSIONS},
        "rationale": draft["rationale"],
    }


def _is_complete(draft: dict[str, object]) -> bool:
    return (
        bool(str(draft.get("annotator_id", "")).strip())
        and all(isinstance(draft.get(dimension), bool) for dimension in JUDGE_DIMENSIONS)
        and not bool(draft.get("issue_flag"))
    )


def _is_issue(draft: dict[str, object]) -> bool:
    return (
        bool(draft.get("issue_flag"))
        and bool(str(draft.get("annotator_id", "")).strip())
        and bool(str(draft.get("issue_note", "")).strip())
    )


def _sqlite_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    return value


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_jsonl(path: Path, payloads: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
