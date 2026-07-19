from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .corpus import Chunk


POLICY_SOURCE_TYPES = frozenset({"integrity_policy", "policy"})


@dataclass(frozen=True)
class EntailmentResult:
    supported: bool
    supporting_chunk_ids: list[str]
    unsupported_claims: list[str]
    confidence: float
    error: str | None = None


class EntailmentVerifier(Protocol):
    model_name: str

    def verify(
        self,
        question: str,
        answer: str,
        chunks: list[Chunk],
    ) -> EntailmentResult:
        ...


def select_relevant_evidence(
    retrieved: list[tuple[Chunk, float]],
    min_score: float | None,
    *,
    policy_min_score: float | None = None,
) -> list[tuple[Chunk, float]]:
    if min_score is None:
        return retrieved
    return [
        (chunk, score)
        for chunk, score in retrieved
        if score >= (
            policy_min_score
            if policy_min_score is not None and chunk.source_type in POLICY_SOURCE_TYPES
            else min_score
        )
    ]
