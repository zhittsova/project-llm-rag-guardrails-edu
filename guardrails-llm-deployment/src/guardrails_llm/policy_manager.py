from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import tempfile
import tomllib
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import tomli_w

from .guard_text import guard_text_candidates
from .guardrail_policy import GuardrailPolicy, guardrail_policy_from_mapping
from .guardrail_runtime import runtime_config_summary


TOP_LEVEL_KEYS = {
    "extends_default",
    "input",
    "context",
    "retrieval",
    "output",
    "messages",
    "coverage_cases",
}
SECTION_KEYS = {
    "input": {"blocking_triggers", "rules", "fuzzy_rules", "similarity_rules"},
    "context": {"rules", "fuzzy_rules"},
    "retrieval": {"allowed_visibility"},
    "output": {"require_citations", "rules", "fuzzy_rules"},
    "messages": {"input_block", "output_block", "ungrounded", "integrity_safe"},
}
RULE_KEYS = {
    "rules": {"trigger", "patterns"},
    "fuzzy_rules": {"trigger", "phrases", "threshold"},
    "similarity_rules": {"trigger", "examples", "threshold"},
}
COVERAGE_CASE_KEYS = {
    "case_id",
    "family",
    "coverage_role",
    "text",
    "expected_triggers",
}
COVERAGE_ROLES = {
    "positive_direct",
    "positive_variant",
    "benign_near_miss",
}


class PolicyManager:
    def __init__(
        self,
        policy_path: Path,
        runtime_config_path: Path,
        *,
        state_dir: Path | None = None,
    ) -> None:
        self.policy_path = policy_path.resolve()
        self.runtime_config_path = runtime_config_path.resolve()
        self.state_dir = (
            state_dir.resolve()
            if state_dir is not None
            else self.policy_path.parent.parent / ".guardrails-policy"
        )
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"policy file does not exist: {self.policy_path}")
        runtime_config_summary(self.runtime_config_path)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.state_dir / "policy-manager.sqlite3"
        self._initialize_database()

    def state(self) -> dict[str, object]:
        draft = self._draft()
        source = self._source_document()
        validation = self.validate(draft)
        source_changed = self._base_sha256() != _file_sha256(self.policy_path)
        return {
            "policy_path": str(self.policy_path),
            "policy_sha256": _file_sha256(self.policy_path),
            "draft_sha256": _document_sha256(draft),
            "dirty": draft != source,
            "source_changed": source_changed,
            "draft": draft,
            "validation": validation,
            "diff": _document_diff(source, draft),
            "runtime": runtime_config_summary(self.runtime_config_path),
            "versions": self.versions(),
        }

    def save_draft(self, document: dict[str, object]) -> dict[str, object]:
        normalized = _json_document(document)
        self._write_draft(normalized)
        return self.state()

    def validate(self, document: dict[str, object]) -> dict[str, object]:
        errors: list[str] = []
        warnings: list[str] = []
        normalized: dict[str, object]
        try:
            normalized = _json_document(document)
        except (TypeError, ValueError) as exc:
            return _validation_payload([str(exc)], warnings, {})

        _validate_shape(normalized, errors)
        if not errors:
            try:
                guardrail_policy_from_mapping(normalized)
            except re.error as exc:
                errors.append(f"invalid regex: {exc}")
            except ValueError as exc:
                errors.append(str(exc))
        _validate_thresholds(normalized, errors)
        coverage = _validate_coverage(normalized, errors)
        if normalized.get("extends_default", True):
            warnings.append(
                "Built-in deterministic guardrails remain active and are read-only here."
            )
        warnings.append(
            "Similarity scores in local simulation use hashing as a preview; "
            "BGE-M3 calibration remains authoritative."
        )
        return _validation_payload(errors, warnings, coverage)

    def publish(self) -> dict[str, object]:
        if self._base_sha256() != _file_sha256(self.policy_path):
            raise ValueError(
                "policy file changed outside the manager; restart or reconcile "
                "the external change before publishing"
            )
        document = self._draft()
        report = self.validate(document)
        if not report["valid"]:
            raise ValueError("; ".join(report["errors"]))
        current = self.policy_path.read_text(encoding="utf-8")
        self._record_version(current, reason="before_publish")
        rendered = _leading_comment_block(current) + _render_document(document)
        _atomic_write(self.policy_path, rendered)
        self._set_base_sha256(_file_sha256(self.policy_path))
        self._write_draft(document)
        return self.state()

    def rollback(self, version_id: int) -> dict[str, object]:
        if self._base_sha256() != _file_sha256(self.policy_path):
            raise ValueError(
                "policy file changed outside the manager; reload it before rollback"
            )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content FROM versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown policy version: {version_id}")
        current = self.policy_path.read_text(encoding="utf-8")
        self._record_version(current, reason=f"before_rollback_{version_id}")
        _atomic_write(self.policy_path, row[0])
        self._set_base_sha256(_file_sha256(self.policy_path))
        self._write_draft(self._source_document())
        return self.state()

    def reload_source(self) -> dict[str, object]:
        document = self._source_document()
        self._write_draft(document)
        self._set_base_sha256(_file_sha256(self.policy_path))
        return self.state()

    def versions(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT version_id, created_at, content_sha256, reason
                FROM versions
                ORDER BY version_id DESC
                """
            ).fetchall()
        return [
            {
                "version_id": row[0],
                "created_at": row[1],
                "sha256": row[2],
                "reason": row[3],
            }
            for row in rows
        ]

    def simulate(self, text: str, *, stage: str = "input") -> dict[str, object]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("simulation text must be a non-empty string")
        if stage not in {"input", "context", "output"}:
            raise ValueError("simulation stage must be input, context, or output")
        document = self._draft()
        report = self.validate(document)
        if not report["valid"]:
            raise ValueError("cannot simulate an invalid draft")
        policy = guardrail_policy_from_mapping(document)
        if stage == "input":
            return _simulate_input(policy, text)
        if stage == "context":
            return _simulate_context(policy, text)
        return _simulate_output(policy, text)

    def run_coverage(self) -> dict[str, object]:
        document = self._draft()
        report = self.validate(document)
        if not report["valid"]:
            raise ValueError("cannot run coverage for an invalid draft")
        policy = guardrail_policy_from_mapping(document)
        results: list[dict[str, object]] = []
        for case in document.get("coverage_cases", []):
            simulation = _simulate_input(policy, case["text"])
            expected = set(case["expected_triggers"])
            actual = set(simulation["triggers"])
            results.append(
                {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "coverage_role": case["coverage_role"],
                    "expected_triggers": sorted(expected),
                    "actual_triggers": sorted(actual),
                    "passed": actual == expected,
                }
            )
        passed = sum(bool(result["passed"]) for result in results)
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results,
            "remote_calls": 0,
            "similarity_provider": "local_hashing_preview",
        }

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS draft (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    document_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS versions (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            existing = connection.execute(
                "SELECT 1 FROM draft WHERE singleton = 1"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO draft VALUES (1, ?, ?)",
                    (json.dumps(self._source_document()), _utc_now()),
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                ("base_sha256", _file_sha256(self.policy_path)),
            )

    def _draft(self) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document_json FROM draft WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("policy draft is missing")
        return json.loads(row[0])

    def _write_draft(self, document: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO draft(singleton, document_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    document_json = excluded.document_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(document, ensure_ascii=False), _utc_now()),
            )

    def _record_version(self, content: str, *, reason: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO versions(created_at, content_sha256, reason, content)
                VALUES (?, ?, ?, ?)
                """,
                (_utc_now(), sha256(content.encode()).hexdigest(), reason, content),
            )

    def _base_sha256(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'base_sha256'"
            ).fetchone()
        if row is None:
            raise RuntimeError("policy draft base hash is missing")
        return row[0]

    def _set_base_sha256(self, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('base_sha256', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (value,),
            )

    def _source_document(self) -> dict[str, object]:
        return tomllib.loads(self.policy_path.read_text(encoding="utf-8"))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=5)


def _simulate_input(policy: GuardrailPolicy, text: str) -> dict[str, object]:
    candidates = guard_text_candidates(text)
    regex = [
        rule.trigger
        for rule in policy.input_rules
        if any(rule.matches(candidate) for candidate in candidates)
    ]
    fuzzy = [
        rule.trigger for rule in policy.input_fuzzy_rules if rule.matches(text)
    ]
    similarity = [
        {
            "trigger": rule.trigger,
            "score": round(score, 6),
            "threshold": rule.threshold,
            "matched": score >= rule.threshold,
        }
        for rule, score in policy._similarity_scores(text)
    ]
    similarity_triggers = [
        item["trigger"] for item in similarity if item["matched"]
    ]
    triggers = _unique(regex + fuzzy + similarity_triggers)
    disposition = "answer"
    if "academic_integrity" in triggers:
        disposition = "redirect"
    if any(trigger in policy.blocking_triggers for trigger in triggers):
        disposition = "block"
    return {
        "stage": "input",
        "text": text,
        "deterministic_triggers": _unique(regex),
        "fuzzy_triggers": _unique(fuzzy),
        "similarity": similarity,
        "triggers": triggers,
        "disposition": disposition,
        "remote_calls": 0,
        "similarity_provider": "local_hashing_preview",
    }


def _simulate_context(policy: GuardrailPolicy, text: str) -> dict[str, object]:
    candidates = guard_text_candidates(text)
    regex = [
        rule.trigger
        for rule in policy.context_rules
        if any(rule.matches(candidate) for candidate in candidates)
    ]
    fuzzy = [
        rule.trigger for rule in policy.context_fuzzy_rules if rule.matches(text)
    ]
    triggers = _unique(regex + fuzzy)
    return {
        "stage": "context",
        "text": text,
        "deterministic_triggers": _unique(regex),
        "fuzzy_triggers": _unique(fuzzy),
        "triggers": triggers,
        "disposition": "block" if triggers else "answer",
        "remote_calls": 0,
    }


def _simulate_output(policy: GuardrailPolicy, text: str) -> dict[str, object]:
    candidates = guard_text_candidates(text)
    regex = [
        rule.trigger
        for rule in policy.output_rules
        if any(rule.matches(candidate) for candidate in candidates)
    ]
    fuzzy = [
        rule.trigger for rule in policy.output_fuzzy_rules if rule.matches(text)
    ]
    triggers = _unique(regex + fuzzy)
    return {
        "stage": "output",
        "text": text,
        "deterministic_triggers": _unique(regex),
        "fuzzy_triggers": _unique(fuzzy),
        "triggers": triggers,
        "disposition": "block" if triggers else "answer",
        "require_citations": policy.require_citations,
        "remote_calls": 0,
    }


def _validate_shape(document: dict[str, object], errors: list[str]) -> None:
    unknown = sorted(set(document) - TOP_LEVEL_KEYS)
    if unknown:
        errors.append(f"unknown policy keys: {', '.join(unknown)}")
    for section, allowed in SECTION_KEYS.items():
        value = document.get(section, {})
        if not isinstance(value, dict):
            errors.append(f"{section} must be a table")
            continue
        section_unknown = sorted(set(value) - allowed)
        if section_unknown:
            errors.append(
                f"unknown {section} keys: {', '.join(section_unknown)}"
            )
        for rule_type, allowed_rule_keys in RULE_KEYS.items():
            rules = value.get(rule_type, [])
            if not isinstance(rules, list):
                continue
            for index, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    continue
                unknown_rule_keys = sorted(set(rule) - allowed_rule_keys)
                if unknown_rule_keys:
                    errors.append(
                        f"unknown {section}.{rule_type}[{index}] keys: "
                        f"{', '.join(unknown_rule_keys)}"
                    )
    coverage_cases = document.get("coverage_cases", [])
    if isinstance(coverage_cases, list):
        for index, case in enumerate(coverage_cases):
            if not isinstance(case, dict):
                continue
            unknown_case_keys = sorted(set(case) - COVERAGE_CASE_KEYS)
            if unknown_case_keys:
                errors.append(
                    f"unknown coverage_cases[{index}] keys: "
                    f"{', '.join(unknown_case_keys)}"
                )


def _validate_thresholds(document: dict[str, object], errors: list[str]) -> None:
    for section_name in ("input", "context", "output"):
        section = document.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for rule_type in ("fuzzy_rules", "similarity_rules"):
            rules = section.get(rule_type, [])
            if not isinstance(rules, list):
                continue
            for index, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    continue
                threshold = rule.get("threshold")
                if not isinstance(threshold, int | float) or not 0 < threshold <= 1:
                    errors.append(
                        f"{section_name}.{rule_type}[{index}].threshold "
                        "must be > 0 and <= 1"
                    )


def _validate_coverage(
    document: dict[str, object],
    errors: list[str],
) -> dict[str, object]:
    families = _input_families(document)
    raw_cases = document.get("coverage_cases", [])
    if not isinstance(raw_cases, list):
        errors.append("coverage_cases must be a list")
        return {}
    roles_by_family: dict[str, set[str]] = {family: set() for family in families}
    ids: list[str] = []
    for index, raw_case in enumerate(raw_cases):
        label = f"coverage_cases[{index}]"
        if not isinstance(raw_case, dict):
            errors.append(f"{label} must be a table")
            continue
        case_id = raw_case.get("case_id")
        family = raw_case.get("family")
        role = raw_case.get("coverage_role")
        text = raw_case.get("text")
        expected = raw_case.get("expected_triggers")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{label}.case_id must be a non-empty string")
        else:
            ids.append(case_id)
        if family not in families:
            errors.append(f"{label}.family must name an input rule trigger")
            continue
        if role not in COVERAGE_ROLES:
            errors.append(f"{label}.coverage_role is invalid")
            continue
        if not isinstance(text, str) or not text:
            errors.append(f"{label}.text must be a non-empty string")
        if not isinstance(expected, list) or not all(
            isinstance(item, str) and item for item in expected
        ):
            errors.append(f"{label}.expected_triggers must be a string list")
        elif role == "benign_near_miss" and family in expected:
            errors.append(f"{label} benign near-miss must not expect {family}")
        elif role != "benign_near_miss" and family not in expected:
            errors.append(f"{label} positive case must expect {family}")
        roles_by_family[family].add(role)
    duplicates = sorted(
        case_id for case_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate coverage case IDs: {', '.join(duplicates)}")
    for family, roles in roles_by_family.items():
        for missing in sorted(COVERAGE_ROLES - roles):
            errors.append(f"{family} is missing coverage role {missing}")
    return {
        "families": sorted(families),
        "roles_by_family": {
            family: sorted(roles) for family, roles in sorted(roles_by_family.items())
        },
        "case_count": len(raw_cases),
    }


def _input_families(document: dict[str, object]) -> set[str]:
    section = document.get("input", {})
    if not isinstance(section, dict):
        return set()
    families: set[str] = set()
    for rule_type in ("rules", "fuzzy_rules", "similarity_rules"):
        rules = section.get(rule_type, [])
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if isinstance(rule, dict) and isinstance(rule.get("trigger"), str):
                families.add(rule["trigger"])
    return families


def _validation_payload(
    errors: list[str],
    warnings: list[str],
    coverage: dict[str, object],
) -> dict[str, object]:
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "coverage": coverage,
    }


def _json_document(document: dict[str, object]) -> dict[str, object]:
    if not isinstance(document, dict):
        raise ValueError("policy document must be an object")
    return json.loads(json.dumps(document, ensure_ascii=False))


def _render_document(document: dict[str, object]) -> str:
    return tomli_w.dumps(document, multiline_strings=True)


def _leading_comment_block(content: str) -> str:
    lines = content.splitlines(keepends=True)
    prefix: list[str] = []
    saw_comment = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            saw_comment = True
            prefix.append(line)
            continue
        if not stripped and saw_comment:
            prefix.append(line)
            continue
        break
    return "".join(prefix)


def _document_diff(
    source: dict[str, object],
    draft: dict[str, object],
) -> str:
    return "".join(
        difflib.unified_diff(
            _render_document(source).splitlines(keepends=True),
            _render_document(draft).splitlines(keepends=True),
            fromfile="published-policy.toml",
            tofile="draft-policy.toml",
        )
    )


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _document_sha256(document: dict[str, object]) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))
