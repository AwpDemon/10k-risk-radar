"""Paths and runtime settings.

Every setting can be overridden with an environment variable so the pipeline runs
against any OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp server, OpenAI, ...).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("RISKRADAR_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
CACHE = DATA / "cache"          # raw EDGAR responses (gitignored)
INTERIM = DATA / "interim"      # parquet tables built by the pipeline (gitignored)
SAMPLE = DATA / "sample"        # small committed sample
GOLD = DATA / "gold"            # hand-labeled evaluation data (committed)
RESULTS = DATA / "results"      # metrics + predictions (committed)
DOCS = ROOT / "docs"
CONFIGS = ROOT / "configs"

# SEC fair-access policy: identify yourself, max 10 requests/second.
SEC_USER_AGENT = os.environ.get(
    "RISKRADAR_SEC_USER_AGENT", "AwpDemon portfolio aliaskari.georgia@gmail.com"
)
SEC_MAX_RPS = float(os.environ.get("RISKRADAR_SEC_MAX_RPS", "5"))


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = field(
        default_factory=lambda: os.environ.get("RISKRADAR_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    )
    model: str = field(default_factory=lambda: os.environ.get("RISKRADAR_LLM_MODEL", "qwen2.5:7b-instruct"))
    api_key: str = field(default_factory=lambda: os.environ.get("RISKRADAR_LLM_API_KEY", "ollama"))
    temperature: float = 0.0
    timeout_s: float = 180.0
    max_retries: int = 3


@dataclass(frozen=True)
class EmbedSettings:
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "RISKRADAR_EMBED_BASE_URL", os.environ.get("RISKRADAR_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        )
    )
    model: str = field(default_factory=lambda: os.environ.get("RISKRADAR_EMBED_MODEL", "nomic-embed-text"))
    api_key: str = field(default_factory=lambda: os.environ.get("RISKRADAR_LLM_API_KEY", "ollama"))
    max_chars: int = 4000
    batch_size: int = 32


def ensure_dirs() -> None:
    for p in (CACHE, INTERIM, SAMPLE, GOLD, RESULTS, DOCS):
        p.mkdir(parents=True, exist_ok=True)
