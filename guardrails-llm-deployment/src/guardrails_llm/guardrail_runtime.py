from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOP_LEVEL_KEYS = {
    "schema_version",
    "profile",
    "endpoint_host",
    "course_id",
    "models",
    "retrieval",
    "classifier",
    "paths",
}


@dataclass(frozen=True)
class RuntimeModels:
    embedding: str
    answer: str
    classifier: str
    entailment: str
    judge: str


@dataclass(frozen=True)
class RuntimeRetrieval:
    top_k: int
    evidence_min_score: float
    policy_context_top_k: int
    policy_context_min_score: float
    entailment_min_confidence: float


@dataclass(frozen=True)
class RuntimeClassifier:
    strategy: str
    min_confidence: float


@dataclass(frozen=True)
class RuntimePaths:
    corpus: Path
    policy: Path
    index: Path
    embedding_cache: Path

    def resolve(self, project_root: Path = PROJECT_ROOT) -> RuntimePaths:
        return RuntimePaths(
            corpus=project_root / self.corpus,
            policy=project_root / self.policy,
            index=project_root / self.index,
            embedding_cache=project_root / self.embedding_cache,
        )


@dataclass(frozen=True)
class GuardrailRuntimeConfig:
    schema_version: int
    profile: str
    endpoint_host: str
    course_id: str
    models: RuntimeModels
    retrieval: RuntimeRetrieval
    classifier: RuntimeClassifier
    paths: RuntimePaths


def default_inhouse_runtime_path() -> Path:
    return PROJECT_ROOT / "data" / "guardrail_runtime_inhouse.toml"


def runtime_config_sha256(path: Path | None = None) -> str:
    target = path or default_inhouse_runtime_path()
    return sha256(target.read_bytes()).hexdigest()


def runtime_config_summary(path: Path | None = None) -> dict[str, object]:
    target = path or default_inhouse_runtime_path()
    config = load_guardrail_runtime_config(target)
    return {
        "path": str(target),
        "sha256": runtime_config_sha256(target),
        "schema_version": config.schema_version,
        "profile": config.profile,
        "endpoint_host": config.endpoint_host,
        "course_id": config.course_id,
        "models": {
            "embedding": config.models.embedding,
            "answer": config.models.answer,
            "classifier": config.models.classifier,
            "entailment": config.models.entailment,
            "judge": config.models.judge,
        },
        "thresholds": {
            "evidence_min_score": config.retrieval.evidence_min_score,
            "policy_context_min_score": config.retrieval.policy_context_min_score,
            "entailment_min_confidence": (
                config.retrieval.entailment_min_confidence
            ),
            "classifier_min_confidence": config.classifier.min_confidence,
        },
        "retrieval": {
            "top_k": config.retrieval.top_k,
            "policy_context_top_k": config.retrieval.policy_context_top_k,
        },
        "paths": {
            "corpus": str(config.paths.corpus),
            "policy": str(config.paths.policy),
            "index": str(config.paths.index),
            "embedding_cache": str(config.paths.embedding_cache),
        },
    }


def load_guardrail_runtime_config(path: Path) -> GuardrailRuntimeConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    _reject_unknown(data, TOP_LEVEL_KEYS, "runtime config")

    schema_version = _as_int(data.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be 1")
    profile = _as_string(data.get("profile"), "profile")
    if profile != "inhouse":
        raise ValueError("profile must be inhouse")

    models_data = _as_table(data.get("models"), "models")
    _reject_unknown(
        models_data,
        {"embedding", "answer", "classifier", "entailment", "judge"},
        "models",
    )
    models = RuntimeModels(
        embedding=_as_string(models_data.get("embedding"), "models.embedding"),
        answer=_as_string(models_data.get("answer"), "models.answer"),
        classifier=_as_string(models_data.get("classifier"), "models.classifier"),
        entailment=_as_string(models_data.get("entailment"), "models.entailment"),
        judge=_as_string(models_data.get("judge"), "models.judge"),
    )

    retrieval_data = _as_table(data.get("retrieval"), "retrieval")
    _reject_unknown(
        retrieval_data,
        {
            "top_k",
            "evidence_min_score",
            "policy_context_top_k",
            "policy_context_min_score",
            "entailment_min_confidence",
        },
        "retrieval",
    )
    retrieval = RuntimeRetrieval(
        top_k=_positive_int(retrieval_data.get("top_k"), "retrieval.top_k"),
        evidence_min_score=_unit_float(
            retrieval_data.get("evidence_min_score"),
            "retrieval.evidence_min_score",
        ),
        policy_context_top_k=_non_negative_int(
            retrieval_data.get("policy_context_top_k"),
            "retrieval.policy_context_top_k",
        ),
        policy_context_min_score=_unit_float(
            retrieval_data.get("policy_context_min_score"),
            "retrieval.policy_context_min_score",
        ),
        entailment_min_confidence=_unit_float(
            retrieval_data.get("entailment_min_confidence"),
            "retrieval.entailment_min_confidence",
        ),
    )

    classifier_data = _as_table(data.get("classifier"), "classifier")
    _reject_unknown(
        classifier_data,
        {"strategy", "min_confidence"},
        "classifier",
    )
    strategy = _as_string(classifier_data.get("strategy"), "classifier.strategy")
    if strategy not in {"always", "ambiguous"}:
        raise ValueError("classifier.strategy must be always or ambiguous")
    classifier = RuntimeClassifier(
        strategy=strategy,
        min_confidence=_unit_float(
            classifier_data.get("min_confidence"),
            "classifier.min_confidence",
        ),
    )

    paths_data = _as_table(data.get("paths"), "paths")
    _reject_unknown(
        paths_data,
        {"corpus", "policy", "index", "embedding_cache"},
        "paths",
    )
    paths = RuntimePaths(
        corpus=_safe_relative_path(paths_data.get("corpus"), "paths.corpus"),
        policy=_safe_relative_path(paths_data.get("policy"), "paths.policy"),
        index=_safe_relative_path(paths_data.get("index"), "paths.index"),
        embedding_cache=_safe_relative_path(
            paths_data.get("embedding_cache"),
            "paths.embedding_cache",
        ),
    )

    return GuardrailRuntimeConfig(
        schema_version=schema_version,
        profile=profile,
        endpoint_host=_as_string(data.get("endpoint_host"), "endpoint_host"),
        course_id=_as_string(data.get("course_id"), "course_id"),
        models=models,
        retrieval=retrieval,
        classifier=classifier,
        paths=paths,
    )


def _reject_unknown(
    data: dict[str, object],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} keys: {', '.join(unknown)}")


def _as_table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table")
    return value


def _as_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _positive_int(value: object, label: str) -> int:
    parsed = _as_int(value, label)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _non_negative_int(value: object, label: str) -> int:
    parsed = _as_int(value, label)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _unit_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return parsed


def _safe_relative_path(value: object, label: str) -> Path:
    path = Path(_as_string(value, label))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay within the project root")
    return path
