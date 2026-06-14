from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


COURSE_ID = "python-intro"


@dataclass(frozen=True)
class CourseCorpusStats:
    source_dir: Path
    output_path: Path
    documents: int


def default_course_source_path() -> Path:
    return Path(__file__).resolve().parents[3] / "course_corpus" / "datainmd"


def default_course_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "python_course_docs.jsonl"


def normalize_course_corpus(
    source_dir: Path,
    output_path: Path,
    *,
    course_id: str = COURSE_ID,
) -> CourseCorpusStats:
    documents = _documents_from_source(source_dir, course_id)
    if not documents:
        raise ValueError(f"No course corpus documents found under {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, sort_keys=True) + "\n")

    return CourseCorpusStats(
        source_dir=source_dir,
        output_path=output_path,
        documents=len(documents),
    )


def _documents_from_source(source_dir: Path, course_id: str) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    policy_path = source_dir / "course_policy.md"
    integrity_path = source_dir / "academic_integrity.md"
    lectures_dir = source_dir / "lecturesmd"

    if policy_path.exists():
        documents.append(
            _document(
                policy_path,
                doc_id="course-policy",
                course_id=course_id,
                title="Course Policy: Intro to CS and Programming with Python",
                source_type="policy",
                source_dir=source_dir,
            )
        )
    if integrity_path.exists():
        documents.append(
            _document(
                integrity_path,
                doc_id="academic-integrity",
                course_id=course_id,
                title="Guidelines for Academic Integrity",
                source_type="integrity_policy",
                source_dir=source_dir,
            )
        )

    for path in sorted(lectures_dir.glob("lec*.md")):
        lecture_number = path.stem.removeprefix("lec").lstrip("0") or "0"
        documents.append(
            _document(
                path,
                doc_id=path.stem,
                course_id=course_id,
                title=_title_from_markdown(path) or f"6.100L Lecture {lecture_number}",
                source_type="lecture_note",
                source_dir=source_dir,
            )
        )

    return documents


def _document(
    path: Path,
    *,
    doc_id: str,
    course_id: str,
    title: str,
    source_type: str,
    source_dir: Path,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "course_id": course_id,
        "title": title,
        "visibility": "public",
        "source_type": source_type,
        "source_path": str(path.relative_to(source_dir)),
        "text": path.read_text(encoding="utf-8"),
    }


def _title_from_markdown(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return ""
