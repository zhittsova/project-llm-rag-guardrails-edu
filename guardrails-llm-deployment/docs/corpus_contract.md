# Corpus Contract

Phase 2 expects course material as newline-delimited JSON. Each line is one
document object. Raw PDF, PPTX, DOCX, and source parsing should happen before
handoff for the current Workshop 2 milestone.

## Required Fields

```json
{
  "doc_id": "lecture-01",
  "course_id": "guardrails-101",
  "title": "Lecture 1: RAG Basics",
  "visibility": "public",
  "source_type": "lecture_note",
  "text": "Document text..."
}
```

Rules:

- `doc_id` must be unique in the file.
- `visibility` must be `public`, `private`, or `internal`.
- Required fields must be non-empty strings.
- Blank lines are ignored.

## Optional Metadata

The loader preserves JSON-compatible extra fields as chunk metadata. Preferred
fields for citations and filtering are:

- `source_path`
- `page`
- `slide`
- `section`
- `author`
- `created_at`
- `allowed_audience`

`allowed_audience` should be a list of strings. Other metadata fields may be
strings, numbers, booleans, null, or lists of strings.

## Validation

Run validation before indexing:

```bash
uv run guardrails-llm validate-corpus --corpus data/course_docs.jsonl
```

Validation fails on malformed JSON, missing required fields, duplicate
`doc_id`, unsupported visibility, empty required values, or unsupported
metadata values.
