from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
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
    normalized = SEPARATOR_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def guard_text_candidates(text: str) -> tuple[str, ...]:
    normalized = normalize_guard_text(text)
    if normalized and normalized != text:
        return (text, normalized)
    return (text,)


def compact_guard_text(text: str) -> str:
    return SEPARATOR_RE.sub("", normalize_guard_text(text))


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
    phrase_token_count = len(normalized_phrase.split())
    min_window = max(1, phrase_token_count - 1)
    max_window = min(len(tokens), phrase_token_count + 6)
    for window_size in range(min_window, max_window + 1):
        for start in range(0, len(tokens) - window_size + 1):
            candidate = " ".join(tokens[start : start + window_size])
            if _similarity(candidate, normalized_phrase) >= threshold:
                return True
            if _similarity(compact_guard_text(candidate), compact_phrase) >= threshold:
                return True
    return False


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()
