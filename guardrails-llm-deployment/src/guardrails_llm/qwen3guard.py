from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI

from .model_config import (
    OpenAIModelConfig,
    ensure_openai_api_key,
    ensure_remote_models_allowed,
    openai_client_kwargs,
)


QWEN3GUARD_MODEL = "qwen3guard-gen-4b"
QWEN3GUARD_PARSER_VERSION = "qwen3guard-native-v1"
QWEN3GUARD_MAPPING_VERSION = "qwen3guard-project-mapping-v1"
SEVERITIES = frozenset({"safe", "controversial", "unsafe"})

NATIVE_CATEGORY_TO_PROJECT_LABEL = {
    "jailbreak": "prompt_injection",
    "pii": "pii",
    "personally identifiable information": "pii",
    "violent": "unsafe_request",
    "non-violent illegal acts": "unsafe_request",
    "sexual content or sexual acts": "unsafe_request",
    "suicide & self-harm": "unsafe_request",
    "politically sensitive topics": "unsafe_request",
    "copyright violation": "unsafe_request",
}


@dataclass(frozen=True)
class Qwen3GuardResult:
    severity: str
    categories: tuple[str, ...]
    raw_text: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity}")

    @property
    def requires_intervention(self) -> bool:
        return self.severity != "safe"


def parse_qwen3guard_output(text: str) -> Qwen3GuardResult:
    match = re.search(r"^\s*Safety\s*:\s*([^\r\n]+)", text, re.I | re.M)
    if match is None:
        raise ValueError("Qwen3Guard response is missing Safety")
    severity = match.group(1).strip().lower()
    if severity not in SEVERITIES:
        raise ValueError(
            f"Qwen3Guard response has unknown severity: {severity}"
        )

    category_match = re.search(
        r"^\s*Categories\s*:\s*([^\r\n]+)",
        text,
        re.I | re.M,
    )
    categories = _normalize_categories(
        category_match.group(1) if category_match else ""
    )
    return Qwen3GuardResult(severity, categories, text)


def map_native_category(result: Qwen3GuardResult) -> str | None:
    if result.severity == "safe":
        return "safe"
    mapped = {
        NATIVE_CATEGORY_TO_PROJECT_LABEL[category]
        for category in result.categories
        if category in NATIVE_CATEGORY_TO_PROJECT_LABEL
    }
    return next(iter(mapped)) if len(mapped) == 1 else None


class Qwen3GuardClassifier:
    def __init__(
        self,
        config: OpenAIModelConfig,
        *,
        client: object | None = None,
    ) -> None:
        ensure_remote_models_allowed(config)
        ensure_openai_api_key(config)
        self.model_name = config.classifier_model
        self._client = client or OpenAI(**openai_client_kwargs(config))

    def classify(self, text: str) -> Qwen3GuardResult:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": text}],
            temperature=0,
        )
        return parse_qwen3guard_output(
            response.choices[0].message.content or ""
        )


def _normalize_categories(value: str) -> tuple[str, ...]:
    categories = tuple(
        category.strip().lower()
        for category in value.split(",")
        if category.strip()
        and category.strip().lower() not in {"none", "n/a", "na"}
    )
    return tuple(dict.fromkeys(categories))
