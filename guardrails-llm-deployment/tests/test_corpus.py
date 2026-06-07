import json
from pathlib import Path

import pytest

from guardrails_llm.corpus import chunk_documents, validate_corpus
from guardrails_llm.langchain_rag import langchain_chunk_documents
from guardrails_llm.pipeline import citation_for


def test_validate_corpus_accepts_and_preserves_metadata(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path,
        [
            {
                "doc_id": "lecture-01",
                "course_id": "guardrails-101",
                "title": "Lecture 1",
                "visibility": "public",
                "source_type": "lecture_note",
                "text": "A short lecture note about retrieval augmented generation.",
                "source_path": "Workshop2/lecture-01.pdf",
                "page": 4,
                "slide": 7,
                "section": "RAG basics",
                "allowed_audience": ["student"],
            }
        ],
    )

    documents = validate_corpus(path)
    word_chunks = chunk_documents(documents, max_words=4)
    langchain_chunks = langchain_chunk_documents(documents, chunk_size=80, chunk_overlap=0)

    assert documents[0].metadata["source_path"] == "Workshop2/lecture-01.pdf"
    assert word_chunks[0].metadata["page"] == 4
    assert langchain_chunks[0].metadata["slide"] == 7
    assert citation_for(langchain_chunks[0]) == "Lecture 1 (lecture-01, RAG basics, slide 7, page 4)"


def test_validate_corpus_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed JSON"):
        validate_corpus(path)


def test_validate_corpus_rejects_missing_required_field(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path,
        [
            {
                "doc_id": "lecture-01",
                "course_id": "guardrails-101",
                "title": "Lecture 1",
                "visibility": "public",
                "source_type": "lecture_note",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing required fields: text"):
        validate_corpus(path)


def test_validate_corpus_rejects_duplicate_doc_id(tmp_path: Path) -> None:
    row = {
        "doc_id": "lecture-01",
        "course_id": "guardrails-101",
        "title": "Lecture 1",
        "visibility": "public",
        "source_type": "lecture_note",
        "text": "Course text.",
    }
    path = write_jsonl(tmp_path, [row, row])

    with pytest.raises(ValueError, match="duplicate doc_id 'lecture-01'"):
        validate_corpus(path)


def test_validate_corpus_rejects_invalid_visibility(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path,
        [
            {
                "doc_id": "lecture-01",
                "course_id": "guardrails-101",
                "title": "Lecture 1",
                "visibility": "secret",
                "source_type": "lecture_note",
                "text": "Course text.",
            }
        ],
    )

    with pytest.raises(ValueError, match="visibility must be one of"):
        validate_corpus(path)


def write_jsonl(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "corpus.jsonl"
    content = "\n".join(json.dumps(row) for row in rows)
    path.write_text(f"{content}\n", encoding="utf-8")
    return path
