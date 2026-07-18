from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from urllib.parse import urlparse

from .guardrail_policy import default_policy_path
from .model_config import (
    OpenAIModelConfig,
    load_local_env,
    resolve_openai_base_url,
    resolve_openai_base_url_source,
)


LOCAL_PROFILE = "local"
INHOUSE_PROFILE = "inhouse"
MODEL_PROFILES = (LOCAL_PROFILE, INHOUSE_PROFILE)
INHOUSE_ENDPOINT_HOST = "learning-services4.fokus.fraunhofer.de"
INHOUSE_EMBEDDING_MODEL = "BAAI/bge-m3"
INHOUSE_LLM_MODEL = "Qwen/Qwen3.6-35B-A3B"


class InHouseEndpointError(RuntimeError):
    pass


def apply_model_profile(args: Namespace) -> None:
    profile = getattr(args, "profile", LOCAL_PROFILE)
    if profile == LOCAL_PROFILE:
        return
    if profile != INHOUSE_PROFILE:
        raise ValueError(f"unknown model profile: {profile}")

    _require_inhouse_endpoint(getattr(args, "env_file", None))
    _set_if_present(args, "retriever", "vector")
    _set_if_present(args, "embedding_provider", "openai")
    _set_if_present(args, "embedding_model", INHOUSE_EMBEDDING_MODEL)
    _set_if_present(args, "guard_embedding_provider", "openai")
    _set_if_present(args, "guard_embedding_model", INHOUSE_EMBEDDING_MODEL)
    _set_if_present(args, "generator", "openai")
    _set_if_present(args, "answer_model", INHOUSE_LLM_MODEL)
    _set_if_present(args, "guard_classifier", "openai")
    _set_if_present(args, "classifier_model", INHOUSE_LLM_MODEL)
    _set_if_present(args, "entailment_verifier", "openai")
    _set_if_present(args, "entailment_model", INHOUSE_LLM_MODEL)
    _set_if_present(args, "judge_model", INHOUSE_LLM_MODEL)
    if hasattr(args, "policy") and args.policy is None:
        args.policy = default_policy_path().with_name("guardrail_policy_bge_m3.toml")


def model_profile_summary(
    profile: str,
    env_file: Path | None = None,
) -> dict[str, object]:
    if profile == LOCAL_PROFILE:
        return {
            "profile": LOCAL_PROFILE,
            "remote_models": False,
            "embedding_provider": "hashing",
            "embedding_model": "hashing-blake2b-384",
            "generator": "extractive",
            "classifier": "none",
            "entailment_verifier": "none",
        }
    if profile != INHOUSE_PROFILE:
        raise ValueError(f"unknown model profile: {profile}")

    config, host = _require_inhouse_endpoint(env_file)
    return {
        "profile": INHOUSE_PROFILE,
        "remote_models": True,
        "endpoint_host": host,
        "endpoint_env": resolve_openai_base_url_source(config),
        "embedding_provider": "openai_compatible",
        "embedding_model": INHOUSE_EMBEDDING_MODEL,
        "answer_model": INHOUSE_LLM_MODEL,
        "classifier_model": INHOUSE_LLM_MODEL,
        "entailment_model": INHOUSE_LLM_MODEL,
        "api_key_present": _api_key_present(config),
        "remote_calls_require_explicit_allowance": True,
        "retrieval_evidence_threshold": "unfrozen",
    }


def _require_inhouse_endpoint(
    env_file: Path | None,
) -> tuple[OpenAIModelConfig, str]:
    config = OpenAIModelConfig(env_file=env_file)
    load_local_env(config.env_file)
    url = resolve_openai_base_url(config)
    host = urlparse(url).hostname if url else None
    if host != INHOUSE_ENDPOINT_HOST:
        actual = host or "not configured"
        raise InHouseEndpointError(
            "The inhouse profile requires the Fraunhofer endpoint host "
            f"{INHOUSE_ENDPOINT_HOST}; resolved host: {actual}."
        )
    return config, host


def _api_key_present(config: OpenAIModelConfig) -> bool:
    import os

    return bool(os.getenv(config.api_key_env))


def _set_if_present(args: Namespace, name: str, value: object) -> None:
    if hasattr(args, name):
        setattr(args, name, value)
