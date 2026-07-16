from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_ANSWER_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_CLASSIFIER_MODEL = "gpt-5.4-nano"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.4-nano"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_URL_ALIAS_ENV = "OPENAI_API_URL"


class RemoteModelsNotAllowedError(RuntimeError):
    pass


class MissingModelCredentialError(RuntimeError):
    pass


class RemoteModelCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIModelConfig:
    api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV
    base_url_env: str = DEFAULT_OPENAI_BASE_URL_ENV
    api_url_alias_env: str = OPENAI_API_URL_ALIAS_ENV
    embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    answer_model: str = DEFAULT_OPENAI_ANSWER_MODEL
    classifier_model: str = DEFAULT_OPENAI_CLASSIFIER_MODEL
    judge_model: str = DEFAULT_OPENAI_JUDGE_MODEL
    env_file: Path | None = None
    allow_remote_models: bool = False

    @property
    def resolved_env_file(self) -> Path:
        return self.env_file or default_env_path()


def default_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def load_local_env(env_file: Path | None = None) -> Path:
    resolved = env_file or default_env_path()
    if resolved.exists():
        load_dotenv(resolved, override=False)
    return resolved


def openai_config_summary(env_file: Path | None = None) -> dict[str, object]:
    config = OpenAIModelConfig(env_file=env_file)
    resolved = load_local_env(config.env_file)
    base_url = resolve_openai_base_url(config)
    base_url_source = resolve_openai_base_url_source(config)
    return {
        "provider": "openai",
        "env_file": str(resolved),
        "env_file_exists": resolved.exists(),
        "api_key_env": config.api_key_env,
        "api_key_present": bool(os.getenv(config.api_key_env)),
        "base_url_env": config.base_url_env,
        "api_url_alias_env": config.api_url_alias_env,
        "base_url_present": bool(base_url),
        "base_url_source": base_url_source,
        "base_url_host": _host_from_url(base_url),
        "uses_chat_completions": should_use_chat_completions(config),
        "embedding_model": DEFAULT_OPENAI_EMBEDDING_MODEL,
        "answer_model": DEFAULT_OPENAI_ANSWER_MODEL,
        "classifier_model": DEFAULT_OPENAI_CLASSIFIER_MODEL,
        "judge_model": DEFAULT_OPENAI_JUDGE_MODEL,
    }


def ensure_remote_models_allowed(config: OpenAIModelConfig) -> None:
    if not config.allow_remote_models:
        raise RemoteModelsNotAllowedError(
            "Remote model calls are disabled. Re-run with --allow-remote-models "
            "only after approving API usage for this command."
        )


def ensure_openai_api_key(config: OpenAIModelConfig) -> None:
    load_local_env(config.env_file)
    if not os.getenv(config.api_key_env):
        raise MissingModelCredentialError(
            f"{config.api_key_env} is not configured. Set it in the shell environment "
            f"or in {config.resolved_env_file}."
        )


def resolve_openai_base_url(config: OpenAIModelConfig) -> str | None:
    load_local_env(config.env_file)
    return _clean_env_value(os.getenv(config.base_url_env)) or _clean_env_value(
        os.getenv(config.api_url_alias_env)
    )


def resolve_openai_base_url_source(config: OpenAIModelConfig) -> str | None:
    load_local_env(config.env_file)
    if _clean_env_value(os.getenv(config.base_url_env)):
        return config.base_url_env
    if _clean_env_value(os.getenv(config.api_url_alias_env)):
        return config.api_url_alias_env
    return None


def should_use_chat_completions(config: OpenAIModelConfig) -> bool:
    return resolve_openai_base_url(config) is not None


def openai_client_kwargs(config: OpenAIModelConfig) -> dict[str, str]:
    load_local_env(config.env_file)
    kwargs = {"api_key": os.getenv(config.api_key_env, "")}
    base_url = resolve_openai_base_url(config)
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip("\"'")
    return value or None


def _host_from_url(url: str | None) -> str | None:
    if not url:
        return None
    return urlparse(url).hostname
