"""
ai_config.py - read-only bridge to the SHARED configuration.

Member B owns `config.py`. This module never redefines or writes shared
configuration. Every setting is resolved in this order:

    1. attribute of the shared ``config`` module (if it exists and defines it)
    2. environment variable with the same name
    3. the default documented in the Member A spec

Only the names listed in ``DEFAULTS`` are read, so if Member B uses a
different spelling for any of them, change it HERE in one place (or add an
alias in ``ALIASES``) instead of touching the other modules.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent

DEFAULTS: dict[str, Any] = {
    # --- Ollama / models ---------------------------------------------------
    "OLLAMA_URL": "http://localhost:11434",
    "CHAT_MODEL": "qwen2.5:3b",              # College Info (always local)
    "EMBED_MODEL": "bge-m3",
    "STUDY_MODEL": "qwen2.5-coder:3b",       # Study Helper (local default)
    "OLLAMA_TIMEOUT": 180,                   # seconds, chat/generate
    "OLLAMA_EMBED_TIMEOUT": 120,
    # --- RAG ---------------------------------------------------------------
    "TOP_K": 4,
    "MAX_DISTANCE": 0.65,
    "TEMPERATURE": 0.1,
    "MAX_QUESTION_CHARS": 500,
    "FALLBACK_ANSWER": "I'm not sure, please contact the office.",
    # --- Chunking / embedding ---------------------------------------------
    "CHUNK_WORDS": 200,
    "CHUNK_OVERLAP": 40,
    "EMBED_BATCH_SIZE": 16,
    # --- Storage -----------------------------------------------------------
    "DATA_DIR": str(BASE_DIR / "data"),
    "CHROMA_DIR": str(BASE_DIR / "data" / "chroma"),
    "COLLECTION_NAME": "sjcit_pages",
    "DEMO_DATA_DIR": str(BASE_DIR / "data_demo"),
    "DEMO_CHROMA_DIR": str(BASE_DIR / "data_demo" / "chroma"),
    "DEMO_COLLECTION_NAME": "sjcit_demo_pages",
    "PAGES_FILE": "pages.jsonl",
    # --- Crawler -----------------------------------------------------------
    "SITE_URL": "",                          # e.g. https://www.example-college.edu/
    "ALLOWED_PATHS": "/",                    # comma separated path prefixes
    "CRAWL_DELAY": 1.0,
    "MAX_PAGES": 300,
    "CRAWL_TIMEOUT": 20,
    "CRAWL_MAX_BYTES": 15_000_000,
    "CRAWL_USER_AGENT": "CampusAIHubBot/1.0 (+authorized college crawler)",
    "CRAWL_AUTHORIZED": "false",             # must be "true" for production crawls
    "MIN_PAGE_WORDS": 30,
    # --- Study Helper ------------------------------------------------------
    "STUDY_PROVIDER": "ollama",              # "ollama" | "anthropic"
    "STUDY_CLOUD_ENABLED": "false",          # must be "true" to allow anthropic
    "ANTHROPIC_API_KEY": "",
    "STUDY_ANTHROPIC_MODEL": "claude-sonnet-5-5",
    "STUDY_MAX_TURNS": 6,
    "STUDY_MAX_HISTORY_CHARS": 4000,
    "STUDY_MAX_QUESTION_CHARS": 4000,
    "STUDY_MAX_PARALLEL": 2,
    "STUDY_MAX_TOKENS": 1200,
}

# Alternative spellings Member B might already use in config.py.
ALIASES: dict[str, tuple[str, ...]] = {
    "OLLAMA_URL": ("OLLAMA_HOST", "OLLAMA_BASE_URL"),
    "CHAT_MODEL": ("COLLEGE_MODEL", "LLM_MODEL"),
    "EMBED_MODEL": ("EMBEDDING_MODEL",),
    "SITE_URL": ("COLLEGE_URL", "COLLEGE_SITE", "COLLEGE_WEBSITE"),
}

_shared = None
_shared_loaded = False


def _load_shared():
    global _shared, _shared_loaded
    if not _shared_loaded:
        _shared_loaded = True
        try:
            _shared = importlib.import_module("config")
        except Exception:  # config.py missing or broken: fall back to env/defaults
            _shared = None
    return _shared


def _coerce(raw: Any, default: Any) -> Any:
    if isinstance(raw, str) and not isinstance(default, str):
        try:
            if isinstance(default, bool):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(default, int):
                return int(raw)
            if isinstance(default, float):
                return float(raw)
        except ValueError:
            return default
    return raw


def get(name: str, default: Any = None) -> Any:
    """Resolve one setting (shared config -> env -> documented default)."""
    fallback = DEFAULTS.get(name, default) if default is None else default
    names = (name,) + ALIASES.get(name, ())
    shared = _load_shared()
    for n in names:
        if shared is not None and hasattr(shared, n):
            return _coerce(getattr(shared, n), fallback)
    for n in names:
        if n in os.environ and os.environ[n] != "":
            return _coerce(os.environ[n], fallback)
    return fallback


def get_bool(name: str) -> bool:
    return str(get(name)).strip().lower() in {"1", "true", "yes", "on"}


def get_list(name: str) -> list[str]:
    value = get(name)
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value or "").split(",") if p.strip()]


def profile_paths(profile: str = "production") -> dict[str, Path]:
    """Return storage locations for a profile. Demo and production NEVER share."""
    if profile == "demo":
        data_dir = Path(get("DEMO_DATA_DIR"))
        chroma_dir = Path(get("DEMO_CHROMA_DIR"))
        collection = str(get("DEMO_COLLECTION_NAME"))
    elif profile == "production":
        data_dir = Path(get("DATA_DIR"))
        chroma_dir = Path(get("CHROMA_DIR"))
        collection = str(get("COLLECTION_NAME"))
    else:
        raise ValueError(f"unknown profile: {profile!r}")
    return {
        "data_dir": data_dir,
        "chroma_dir": chroma_dir,
        "collection": collection,  # type: ignore[dict-item]
        "pages_file": data_dir / str(get("PAGES_FILE")),
    }


def assert_profiles_separate() -> None:
    """Hard guard: demo and production must differ in directory AND collection."""
    prod, demo = profile_paths("production"), profile_paths("demo")
    if Path(prod["chroma_dir"]).resolve() == Path(demo["chroma_dir"]).resolve():
        raise RuntimeError("DEMO_CHROMA_DIR must differ from CHROMA_DIR")
    if prod["collection"] == demo["collection"]:
        raise RuntimeError("DEMO_COLLECTION_NAME must differ from COLLECTION_NAME")
    if Path(prod["data_dir"]).resolve() == Path(demo["data_dir"]).resolve():
        raise RuntimeError("DEMO_DATA_DIR must differ from DATA_DIR")
