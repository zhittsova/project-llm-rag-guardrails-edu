from pathlib import Path

import pytest

from guardrails_llm.corpus import Document, load_documents
from guardrails_llm.retrieval_evaluation import (
    RetrievalCase,
    load_retrieval_cases,
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
