from pathlib import Path

import pytest

from guardrails_llm.corpus import Chunk, Document, load_documents
from guardrails_llm.retrieval_evaluation import (
    RetrievalCase,
    RetrievalResult,
    load_retrieval_cases,
    retrieval_results_to_json,
    run_retrieval_evaluation,
    summarize_retrieval,
    validate_retrieval_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "course_docs.jsonl"
CASES_PATH = ROOT / "data" / "retrieval_cases_milestone3_v1.jsonl"


def _case(**overrides) -> RetrievalCase:
    values = {
        "case_id": "retrieval-001",
        "kind": "relevance",
        "question": "What is retrieval augmented generation?",
        "course_id": "guardrails-101",
        "expected_doc_ids": ["rag-basics"],
        "forbidden_doc_ids": [],
        "allowed_visibility": ["public"],
        "difficulty": "easy",
    }
    values.update(overrides)
    return RetrievalCase(**values)


def test_retrieval_case_accepts_relevance_and_visibility_contracts() -> None:
    relevance = _case()
    visibility = _case(
        case_id="visibility-001",
        kind="visibility",
        expected_doc_ids=[],
        forbidden_doc_ids=["private-roster"],
    )

    assert relevance.expected_doc_ids == ["rag-basics"]
    assert visibility.forbidden_doc_ids == ["private-roster"]


@pytest.mark.parametrize("kind", ["answer", "private", ""])
def test_retrieval_case_rejects_unknown_kind(kind: str) -> None:
    with pytest.raises(ValueError, match="kind"):
        _case(kind=kind)


@pytest.mark.parametrize("difficulty", ["simple", "advanced", ""])
def test_retrieval_case_rejects_unknown_difficulty(difficulty: str) -> None:
    with pytest.raises(ValueError, match="difficulty"):
        _case(difficulty=difficulty)


def test_retrieval_case_requires_kind_specific_document_ids() -> None:
    with pytest.raises(ValueError, match="expected_doc_ids"):
        _case(expected_doc_ids=[])

    with pytest.raises(ValueError, match="forbidden_doc_ids"):
        _case(kind="visibility", expected_doc_ids=[], forbidden_doc_ids=[])


def test_load_retrieval_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    line = (
        '{"case_id":"duplicate","kind":"relevance","question":"Question?",'
        '"course_id":"guardrails-101","expected_doc_ids":["rag-basics"],'
        '"forbidden_doc_ids":[],"allowed_visibility":["public"],'
        '"difficulty":"easy"}\n'
    )
    path.write_text(line + line, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_retrieval_cases(path)


def test_validate_retrieval_cases_rejects_unknown_or_inaccessible_documents() -> None:
    documents = [
        Document(
            doc_id="public-doc",
            course_id="guardrails-101",
            title="Public",
            visibility="public",
            source_type="lecture_note",
            text="Public material.",
        ),
        Document(
            doc_id="private-doc",
            course_id="guardrails-101",
            title="Private",
            visibility="private",
            source_type="admin_note",
            text="Private material.",
        ),
    ]

    with pytest.raises(ValueError, match="unknown document"):
        validate_retrieval_cases(
            [_case(expected_doc_ids=["missing-doc"])],
            documents,
        )

    with pytest.raises(ValueError, match="not allowed"):
        validate_retrieval_cases(
            [_case(expected_doc_ids=["private-doc"])],
            documents,
        )


def test_frozen_retrieval_dataset_is_valid_and_balanced() -> None:
    cases = load_retrieval_cases(CASES_PATH)
    documents = load_documents(CORPUS_PATH)

    validate_retrieval_cases(cases, documents)

    assert len(cases) == 24
    assert sum(case.kind == "relevance" for case in cases) == 20
    assert sum(case.kind == "visibility" for case in cases) == 4
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.difficulty for case in cases} == {"easy", "medium", "hard"}
    assert {
        doc_id
        for case in cases
        for doc_id in case.expected_doc_ids
    } == {
        "rag-basics",
        "threat-model",
        "guardrail-pipeline",
        "integrity-policy",
        "assignment-policy",
    }
    assert all(
        case.forbidden_doc_ids == ["private-roster"]
        for case in cases
        if case.kind == "visibility"
    )


def _chunk(doc_id: str, index: int = 0, visibility: str = "public") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        course_id="guardrails-101",
        title=doc_id,
        visibility=visibility,
        source_type="lecture_note",
        text=f"Text from {doc_id}.",
    )


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, **kwargs) -> list[tuple[Chunk, float]]:
        self.calls.append({"query": query, **kwargs})
        return [
            (chunk, 1.0 - index / 10)
            for index, chunk in enumerate(self.chunks)
        ]


def test_run_retrieval_evaluation_ranks_unique_documents() -> None:
    retriever = FakeRetriever(
        [_chunk("wrong", 0), _chunk("wrong", 1), _chunk("rag-basics")]
    )

    result = run_retrieval_evaluation(retriever, [_case()])[0]

    assert result.retrieved_doc_ids == ["wrong", "rag-basics"]
    assert result.first_relevant_rank == 2
    assert result.forbidden_hits == []
    assert result.passed
    assert retriever.calls[0]["course_id"] == "guardrails-101"
    assert retriever.calls[0]["allowed_visibility"] == {"public"}


def test_run_retrieval_evaluation_records_visibility_failure() -> None:
    retriever = FakeRetriever(
        [_chunk("private-roster", visibility="private"), _chunk("rag-basics")]
    )
    case = _case(
        kind="visibility",
        expected_doc_ids=[],
        forbidden_doc_ids=["private-roster"],
    )

    result = run_retrieval_evaluation(retriever, [case])[0]

    assert result.first_relevant_rank is None
    assert result.forbidden_hits == ["private-roster"]
    assert not result.passed


def _result(
    case_id: str,
    *,
    kind: str = "relevance",
    difficulty: str = "easy",
    rank: int | None = 1,
    forbidden_hits: list[str] | None = None,
    latency_ms: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        case_id=case_id,
        kind=kind,
        difficulty=difficulty,
        expected_doc_ids=["expected"] if kind == "relevance" else [],
        forbidden_doc_ids=["private"] if kind == "visibility" else [],
        retrieved_doc_ids=[],
        first_relevant_rank=rank if kind == "relevance" else None,
        forbidden_hits=forbidden_hits or [],
        passed=(rank is not None) if kind == "relevance" else not forbidden_hits,
        latency_ms=latency_ms,
    )


def test_summarize_retrieval_uses_hand_calculated_metrics() -> None:
    results = [
        _result("rank-1", rank=1, difficulty="easy", latency_ms=1.0),
        _result("rank-2", rank=2, difficulty="easy", latency_ms=2.0),
        _result("miss", rank=None, difficulty="hard", latency_ms=3.0),
        _result(
            "visibility-pass",
            kind="visibility",
            difficulty="hard",
            latency_ms=4.0,
        ),
        _result(
            "visibility-fail",
            kind="visibility",
            difficulty="medium",
            forbidden_hits=["private"],
            latency_ms=5.0,
        ),
    ]

    summary = summarize_retrieval(results)

    assert summary["total"] == 5
    assert summary["relevance_total"] == 3
    assert summary["visibility_total"] == 2
    assert summary["recall_at_1"] == 0.333
    assert summary["recall_at_3"] == 0.667
    assert summary["mrr"] == 0.5
    assert summary["visibility_filter_success_rate"] == 0.5
    assert summary["avg_latency_ms"] == 3.0
    assert summary["by_difficulty"]["easy"]["recall_at_1"] == 0.5
    assert summary["by_difficulty"]["easy"]["mrr"] == 0.75
    assert summary["by_difficulty"]["hard"]["visibility_filter_success_rate"] == 1.0
    assert summary["by_difficulty"]["medium"]["relevance_total"] == 0


def test_summarize_retrieval_handles_empty_results() -> None:
    assert summarize_retrieval([]) == {
        "total": 0,
        "passed": 0,
        "pass_rate": 0.0,
        "relevance_total": 0,
        "visibility_total": 0,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "mrr": 0.0,
        "visibility_filter_success_rate": 0.0,
        "avg_latency_ms": 0.0,
        "by_difficulty": {},
    }


def test_retrieval_results_serialize_as_json() -> None:
    encoded = retrieval_results_to_json([_result("rank-1")])

    assert '"case_id": "rank-1"' in encoded
    assert '"first_relevant_rank": 1' in encoded
