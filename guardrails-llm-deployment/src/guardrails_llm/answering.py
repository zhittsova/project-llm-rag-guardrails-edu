from __future__ import annotations

from typing import Protocol

from .corpus import Chunk


class AnswerGenerator(Protocol):
    model_name: str

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        ...
