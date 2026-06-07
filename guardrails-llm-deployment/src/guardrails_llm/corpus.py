from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "doc_id",
    "course_id",
    "title",
    "visibility",
    "source_type",
    "text",
}
VISIBILITY_VALUES = {"public", "private", "internal"}

JsonMetadata = str | int | float | bool | None | list[str]


@dataclass(frozen=True)
class Document:
    doc_id: str
    course_id: str
    title: str
    visibility: str
    source_type: str
    text: str
    metadata: dict[str, JsonMetadata] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    course_id: str
    title: str
    visibility: str
    source_type: str
    text: str
    metadata: dict[str, JsonMetadata] = field(default_factory=dict)


def load_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen_doc_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = _load_json_line(line, path, line_number)
            document = document_from_payload(payload, path, line_number)
            if document.doc_id in seen_doc_ids:
                raise ValueError(f"Invalid document at {path}:{line_number}: duplicate doc_id '{document.doc_id}'")
            seen_doc_ids.add(document.doc_id)
            documents.append(document)
    return documents


def validate_corpus(path: Path) -> list[Document]:
    return load_documents(path)


def document_from_payload(payload: Any, path: Path, line_number: int) -> Document:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid document at {path}:{line_number}: expected a JSON object")

    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"Invalid document at {path}:{line_number}: missing required fields: {fields}")

    required_values = {field_name: payload[field_name] for field_name in REQUIRED_FIELDS}
    for field_name, value in required_values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid document at {path}:{line_number}: '{field_name}' must be a non-empty string"
            )

    visibility = required_values["visibility"]
    if visibility not in VISIBILITY_VALUES:
        allowed = ", ".join(sorted(VISIBILITY_VALUES))
        raise ValueError(
            f"Invalid document at {path}:{line_number}: visibility must be one of {allowed}"
        )

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in REQUIRED_FIELDS and _is_json_metadata(value)
    }
    rejected = sorted(
        key for key, value in payload.items() if key not in REQUIRED_FIELDS and not _is_json_metadata(value)
    )
    if rejected:
        fields = ", ".join(rejected)
        raise ValueError(f"Invalid document at {path}:{line_number}: unsupported metadata fields: {fields}")

    return Document(
        doc_id=required_values["doc_id"],
        course_id=required_values["course_id"],
        title=required_values["title"],
        visibility=visibility,
        source_type=required_values["source_type"],
        text=required_values["text"],
        metadata=metadata,
    )


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
                    metadata=dict(document.metadata),
                )
            )
    return chunks


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "course_docs.jsonl"


def _load_json_line(line: str, path: Path, line_number: int) -> Any:
    try:
        return json.loads(line)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid document at {path}:{line_number}: malformed JSON") from exc


def _is_json_metadata(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(isinstance(item, str) for item in value)
    return False
