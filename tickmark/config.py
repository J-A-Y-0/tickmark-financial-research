"""Settings, read once from the environment (and a local .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional: a local .env for development
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value else default


def _int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # Models. "offline" needs no keys: heuristic planner, keyword router, extractive answers.
    llm_provider: str = "offline"  # offline | openai | anthropic
    answer_model: str = ""
    fast_model: str = ""  # planning and the LLM router
    embedding_provider: str = "hashing"  # hashing | openai
    embedding_model: str = "text-embedding-3-small"

    router: str = "keyword"  # keyword | llm | jev | cascade
    jev_model: str = "jev-latest"
    cascade_threshold: float = 0.8
    verifier: str = "lexical"  # lexical | jev

    search_provider: str = "none"  # tavily | serpapi | none
    search_fallback: str = "none"

    database_path: Path = field(default_factory=lambda: ROOT / "var" / "tickmark.sqlite3")
    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    # Bounds and spend controls
    quick_timeout_s: float = 90.0
    deep_timeout_s: float = 240.0
    deep_daily_cap: int = 5
    daily_budget_usd: float = 5.0
    max_upload_mb: int = 25

    # Semantic cache
    cache_similarity: float = 0.92
    cache_ttl_hours: float = 24.0

    @property
    def offline(self) -> bool:
        return self.llm_provider == "offline"


DEFAULT_MODELS = {
    "openai": ("gpt-5.5", "gpt-5.4-mini"),
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5"),
    "offline": ("extractive", "heuristic"),
}


def load_settings() -> Settings:
    provider = _env("LLM_PROVIDER", "offline").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {provider!r}")
    answer_default, fast_default = DEFAULT_MODELS[provider]
    default_router = "keyword" if provider == "offline" else "llm"
    db = _env("DATABASE_PATH")
    return Settings(
        llm_provider=provider,
        answer_model=_env("ANSWER_MODEL", answer_default),
        fast_model=_env("FAST_MODEL", fast_default),
        embedding_provider=_env("EMBEDDING_PROVIDER", "hashing").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        router=_env("ROUTER", default_router).lower(),
        jev_model=_env("JEV_MODEL", "jev-latest"),
        cascade_threshold=_float("CASCADE_THRESHOLD", 0.8),
        verifier=_env("VERIFIER", "lexical").lower(),
        search_provider=_env("SEARCH_PROVIDER", "none").lower(),
        search_fallback=_env("SEARCH_FALLBACK", "none").lower(),
        database_path=Path(db) if db else ROOT / "var" / "tickmark.sqlite3",
        quick_timeout_s=_float("QUICK_TIMEOUT_S", 90.0),
        deep_timeout_s=_float("DEEP_TIMEOUT_S", 240.0),
        deep_daily_cap=_int("DEEP_DAILY_CAP", 5),
        daily_budget_usd=_float("DAILY_BUDGET_USD", 5.0),
        max_upload_mb=_int("MAX_UPLOAD_MB", 25),
        cache_similarity=_float("CACHE_SIMILARITY", 0.92),
        cache_ttl_hours=_float("CACHE_TTL_HOURS", 24.0),
    )
