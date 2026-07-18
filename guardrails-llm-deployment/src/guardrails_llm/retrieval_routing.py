from __future__ import annotations


ACADEMIC_INTEGRITY_RETRIEVAL_QUERY = (
    "academic integrity graded work complete submissions hints similar examples"
)


def route_retrieval_query(question: str, triggers: set[str]) -> str:
    if "academic_integrity" in triggers:
        return ACADEMIC_INTEGRITY_RETRIEVAL_QUERY
    return question
