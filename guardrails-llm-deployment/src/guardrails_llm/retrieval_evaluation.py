from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .corpus import Chunk, Document, VISIBILITY_VALUES


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


@dataclass(frozen=True)
class RetrievalResult:
    case_id: str
    kind: str
    difficulty: str
    expected_doc_ids: list[str]
    forbidden_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    first_relevant_rank: int | None
    forbidden_hits: list[str]
    passed: bool
    latency_ms: float


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        course_id: str | None = None,
        allowed_visibility: set[str] | None = None,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        ...


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


def run_retrieval_evaluation(
    retriever: Retriever,
    cases: list[RetrievalCase],
    *,
    top_k: int = 3,
) -> list[RetrievalResult]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    results: list[RetrievalResult] = []
    for case in cases:
        started_at = perf_counter()
        matches = retriever.search(
            case.question,
            course_id=case.course_id,
            allowed_visibility=set(case.allowed_visibility),
            top_k=max(top_k * 4, top_k),
        )
        retrieved_doc_ids = _unique_doc_ids(matches, limit=top_k)
        first_relevant_rank = next(
            (
                rank
                for rank, doc_id in enumerate(retrieved_doc_ids, start=1)
                if doc_id in case.expected_doc_ids
            ),
            None,
        )
        forbidden_hits = [
            doc_id
            for doc_id in retrieved_doc_ids
            if doc_id in case.forbidden_doc_ids
        ]
        passed = (
            first_relevant_rank is not None
            if case.kind == "relevance"
            else not forbidden_hits
        )
        results.append(
            RetrievalResult(
                case_id=case.case_id,
                kind=case.kind,
                difficulty=case.difficulty,
                expected_doc_ids=list(case.expected_doc_ids),
                forbidden_doc_ids=list(case.forbidden_doc_ids),
                retrieved_doc_ids=retrieved_doc_ids,
                first_relevant_rank=first_relevant_rank,
                forbidden_hits=forbidden_hits,
                passed=passed,
                latency_ms=(perf_counter() - started_at) * 1000,
            )
        )
    return results


def summarize_retrieval(results: list[RetrievalResult]) -> dict[str, object]:
    summary = _summarize_group(results)
    by_difficulty: dict[str, dict[str, int | float]] = {}
    for difficulty in DIFFICULTIES:
        grouped = [result for result in results if result.difficulty == difficulty]
        if grouped:
            by_difficulty[difficulty] = _summarize_group(grouped)
    return summary | {"by_difficulty": by_difficulty}


def retrieval_results_to_json(results: list[RetrievalResult]) -> str:
    return json.dumps([asdict(result) for result in results], indent=2)


def _summarize_group(results: list[RetrievalResult]) -> dict[str, int | float]:
    total = len(results)
    relevance = [result for result in results if result.kind == "relevance"]
    visibility = [result for result in results if result.kind == "visibility"]
    reciprocal_rank = sum(
        1 / result.first_relevant_rank
        if result.first_relevant_rank is not None
        else 0.0
        for result in relevance
    )
    return {
        "total": total,
        "passed": sum(result.passed for result in results),
        "pass_rate": round(sum(result.passed for result in results) / total, 3)
        if total
        else 0.0,
        "relevance_total": len(relevance),
        "visibility_total": len(visibility),
        "recall_at_1": _rate(
            sum(result.first_relevant_rank == 1 for result in relevance),
            len(relevance),
        ),
        "recall_at_3": _rate(
            sum(
                result.first_relevant_rank is not None
                and result.first_relevant_rank <= 3
                for result in relevance
            ),
            len(relevance),
        ),
        "mrr": round(reciprocal_rank / len(relevance), 3) if relevance else 0.0,
        "visibility_filter_success_rate": _rate(
            sum(not result.forbidden_hits for result in visibility),
            len(visibility),
        ),
        "avg_latency_ms": round(
            sum(result.latency_ms for result in results) / total,
            2,
        )
        if total
        else 0.0,
    }


def _unique_doc_ids(
    matches: list[tuple[Chunk, float]],
    *,
    limit: int,
) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for chunk, _score in matches:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        doc_ids.append(chunk.doc_id)
        if len(doc_ids) == limit:
            break
    return doc_ids


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _validate_string_list(case_id: str, field_name: str, value: object) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{case_id}: {field_name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{case_id}: {field_name} must not contain duplicates")
