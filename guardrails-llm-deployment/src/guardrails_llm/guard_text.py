from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
    }
)


def normalize_guard_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = normalized.casefold().translate(LEET_TRANSLATION)
    normalized = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(normalized.split())


def guard_text_candidates(text: str) -> tuple[str, ...]:
    normalized = normalize_guard_text(text)
    if normalized and normalized != text:
        return (text, normalized)
    return (text,)


def compact_guard_text(text: str) -> str:
    return "".join(character for character in normalize_guard_text(text) if character.isalnum())


def fuzzy_phrase_matches(text: str, phrase: str, *, threshold: float) -> bool:
    normalized_text = normalize_guard_text(text)
    normalized_phrase = normalize_guard_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    if normalized_phrase in normalized_text:
        return True

    compact_text = compact_guard_text(normalized_text)
    compact_phrase = compact_guard_text(normalized_phrase)
    if compact_phrase and compact_phrase in compact_text:
        return True

    tokens = normalized_text.split()
    phrase_tokens = normalized_phrase.split()
    phrase_token_count = len(phrase_tokens)
    min_window = max(1, phrase_token_count - 1)
    max_window = min(len(tokens), phrase_token_count + 6)
    starts = _candidate_starts(tokens, phrase_tokens, min_window, max_window)
    for window_size in range(min_window, max_window + 1):
        for start in starts[window_size]:
            candidate = " ".join(tokens[start : start + window_size])
            if _similarity(candidate, normalized_phrase) >= threshold:
                return True
            compact_candidate = "".join(candidate.split())
            if _similarity(compact_candidate, compact_phrase) >= threshold:
                return True
    return False


def _candidate_starts(
    tokens: list[str],
    phrase_tokens: list[str],
    min_window: int,
    max_window: int,
) -> dict[int, set[int]]:
    starts = {window_size: set() for window_size in range(min_window, max_window + 1)}
    anchor_tokens = {token for token in phrase_tokens if len(token) >= 5}
    anchor_indexes = [index for index, token in enumerate(tokens) if token in anchor_tokens]

    if not anchor_indexes:
        return starts

    for index in anchor_indexes:
        for window_size in starts:
            earliest = max(0, index - window_size + 1)
            latest = min(index, len(tokens) - window_size)
            if latest >= earliest:
                starts[window_size].update(range(earliest, latest + 1))
    return starts


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()
