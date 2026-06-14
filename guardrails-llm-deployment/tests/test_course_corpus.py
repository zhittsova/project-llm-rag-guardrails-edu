import json
from pathlib import Path

from guardrails_llm.corpus import validate_corpus
from guardrails_llm.course_corpus import normalize_course_corpus


def test_normalize_course_corpus_creates_valid_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "datainmd"
    lectures = source / "lecturesmd"
    lectures.mkdir(parents=True)
    (source / "course_policy.md").write_text("# Course Policy\nPython 3 course.", encoding="utf-8")
    (source / "academic_integrity.md").write_text("# Integrity\nDo not submit generated code.", encoding="utf-8")
    (lectures / "lec01.md").write_text("# Lecture 1\nDeclarative knowledge is facts.", encoding="utf-8")
    output = tmp_path / "course.jsonl"

    stats = normalize_course_corpus(source, output, course_id="python-intro")
    documents = validate_corpus(output)

    assert stats.documents == 3
    assert [document.doc_id for document in documents] == ["course-policy", "academic-integrity", "lec01"]
    assert all(document.course_id == "python-intro" for document in documents)
    assert documents[2].metadata["source_path"] == "lecturesmd/lec01.md"


def test_normalized_python_course_fixture_is_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "python_course_docs.jsonl"

    documents = validate_corpus(path)

    assert len(documents) == 28
    assert {document.course_id for document in documents} == {"python-intro"}
    assert "lec01" in {document.doc_id for document in documents}


def test_promptfoo_jsonl_is_not_course_corpus_format(tmp_path: Path) -> None:
    path = tmp_path / "test01.jsonl"
    path.write_text(json.dumps({"description": "test", "vars": {"test_prompt": "hello"}}) + "\n", encoding="utf-8")

    try:
        validate_corpus(path)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("promptfoo-style JSONL should not validate as corpus JSONL")
