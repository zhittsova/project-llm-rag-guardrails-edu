from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .corpus import Chunk


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    answerable: bool | None = None


class AnswerGenerator(Protocol):
    model_name: str

    def generate(self, question: str, chunks: list[Chunk]) -> str | GeneratedAnswer:
        ...


def unpack_generated_answer(value: str | GeneratedAnswer) -> GeneratedAnswer:
    if isinstance(value, GeneratedAnswer):
        return value
    return GeneratedAnswer(text=value)
