from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    course_id: str
    title: str
    visibility: str
    source_type: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    course_id: str
    title: str
    visibility: str
    source_type: str
    text: str


def load_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                documents.append(Document(**payload))
            except TypeError as exc:
                raise ValueError(f"Invalid document at {path}:{line_number}") from exc
    return documents


def chunk_documents(documents: list[Document], max_words: int = 95) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        for index in range(0, len(words), max_words):
            chunk_words = words[index : index + max_words]
            if not chunk_words:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index // max_words}",
                    doc_id=document.doc_id,
                    course_id=document.course_id,
                    title=document.title,
                    visibility=document.visibility,
                    source_type=document.source_type,
                    text=" ".join(chunk_words),
                )
            )
    return chunks


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "course_docs.jsonl"

