from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from .corpus import Document, VISIBILITY_VALUES


RETRIEVAL_KINDS = frozenset({"relevance", "visibility"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    kind: str
    question: str
    course_id: str
    expected_doc_ids: list[str]
    forbidden_doc_ids: list[str]
    allowed_visibility: list[str]
    difficulty: str

    def __post_init__(self) -> None:
        for field_name in ("case_id", "question", "course_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.kind not in RETRIEVAL_KINDS:
            raise ValueError("kind must be relevance or visibility")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError("difficulty must be easy, medium, or hard")
        _validate_string_list(self.case_id, "expected_doc_ids", self.expected_doc_ids)
        _validate_string_list(self.case_id, "forbidden_doc_ids", self.forbidden_doc_ids)
        _validate_string_list(self.case_id, "allowed_visibility", self.allowed_visibility)
        if not self.allowed_visibility:
            raise ValueError(f"{self.case_id}: allowed_visibility must not be empty")
        unknown_visibility = set(self.allowed_visibility) - VISIBILITY_VALUES
        if unknown_visibility:
            raise ValueError(
                f"{self.case_id}: unknown allowed_visibility: "
                f"{', '.join(sorted(unknown_visibility))}"
            )
        if set(self.expected_doc_ids) & set(self.forbidden_doc_ids):
            raise ValueError(
                f"{self.case_id}: expected_doc_ids and forbidden_doc_ids must not overlap"
            )
        if self.kind == "relevance" and not self.expected_doc_ids:
            raise ValueError(f"{self.case_id}: relevance case requires expected_doc_ids")
        if self.kind == "visibility":
            if self.expected_doc_ids:
                raise ValueError(
                    f"{self.case_id}: visibility case must not define expected_doc_ids"
                )
            if not self.forbidden_doc_ids:
                raise ValueError(
                    f"{self.case_id}: visibility case requires forbidden_doc_ids"
                )


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid retrieval case at {path}:{line_number}: malformed JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid retrieval case at {path}:{line_number}: expected an object"
                )
            try:
                case = RetrievalCase(**payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid retrieval case at {path}:{line_number}: {exc}"
                ) from exc
            if case.case_id in seen_ids:
                raise ValueError(
                    f"Invalid retrieval case at {path}:{line_number}: "
                    f"duplicate case_id '{case.case_id}'"
                )
            seen_ids.add(case.case_id)
            cases.append(case)
    return cases


def validate_retrieval_cases(
    cases: list[RetrievalCase],
    documents: list[Document],
) -> None:
    documents_by_id = {document.doc_id: document for document in documents}
    for case in cases:
        for doc_id in case.expected_doc_ids + case.forbidden_doc_ids:
            document = documents_by_id.get(doc_id)
            if document is None:
                raise ValueError(f"{case.case_id}: unknown document '{doc_id}'")
            if document.course_id != case.course_id:
                raise ValueError(
                    f"{case.case_id}: document '{doc_id}' belongs to another course"
                )
        for doc_id in case.expected_doc_ids:
            document = documents_by_id[doc_id]
            if document.visibility not in case.allowed_visibility:
                raise ValueError(
                    f"{case.case_id}: expected document '{doc_id}' is not allowed by "
                    "allowed_visibility"
                )
        for doc_id in case.forbidden_doc_ids:
            document = documents_by_id[doc_id]
            if document.visibility in case.allowed_visibility:
                raise ValueError(
                    f"{case.case_id}: forbidden document '{doc_id}' is allowed by "
                    "allowed_visibility"
                )


def _validate_string_list(case_id: str, field_name: str, value: object) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{case_id}: {field_name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{case_id}: {field_name} must not contain duplicates")
