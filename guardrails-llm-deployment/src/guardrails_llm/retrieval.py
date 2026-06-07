from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import Chunk

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "assistant",
    "be",
    "can",
    "course",
    "do",
    "does",
    "exact",
    "for",
    "in",
    "is",
    "it",
    "me",
    "model",
    "my",
    "of",
    "on",
    "should",
    "the",
    "to",
    "train",
    "used",
    "was",
    "what",
    "which",
    "with",
    "you",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for match in TOKEN_RE.finditer(text)
        if (token := match.group(0).lower()) not in STOPWORDS
    ]


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        doc_freq: Counter[str] = Counter()
        for counts in self._term_counts:
            doc_freq.update(counts.keys())
        total = max(len(chunks), 1)
        self._idf = {
            term: math.log((1 + total) / (1 + freq)) + 1
            for term, freq in doc_freq.items()
        }

    def search(
        self,
        query: str,
        *,
        course_id: str | None = None,
        allowed_visibility: set[str] | None = None,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        query_counts = Counter(tokenize(query))
        scored: list[tuple[Chunk, float]] = []
        for chunk, counts in zip(self._chunks, self._term_counts, strict=True):
            if course_id and chunk.course_id != course_id:
                continue
            if allowed_visibility and chunk.visibility not in allowed_visibility:
                continue
            score = sum(
                query_count * counts.get(term, 0) * self._idf.get(term, 1.0)
                for term, query_count in query_counts.items()
            )
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
