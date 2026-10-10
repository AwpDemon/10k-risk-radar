"""Thin OpenAI-compatible client with a disk cache, JSON parsing, and usage accounting.

Works with any endpoint that speaks the OpenAI Chat Completions / Embeddings API:
Ollama (``http://127.0.0.1:11434/v1``), vLLM, llama.cpp server, LM Studio, or OpenAI itself.
Set ``RISKRADAR_LLM_BASE_URL``, ``RISKRADAR_LLM_MODEL`` and ``RISKRADAR_LLM_API_KEY``.

Every response is cached by a hash of (model, messages, params) so evaluations are
reproducible and re-runs are free.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import config


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


class ChatBackend(Protocol):
    def __call__(self, model: str, messages: list[dict], **params: Any) -> ChatResult: ...


class _Cache:
    """Tiny sqlite key-value cache (safe across threads with a lock)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, k: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def set(self, k: str, v: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, v))
            self._conn.commit()


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class LLMClient:
    def __init__(self, settings: config.LLMSettings | None = None, cache_path: Path | None = None,
                 backend: ChatBackend | None = None):
        self.settings = settings or config.LLMSettings()
        self.cache = _Cache(cache_path or config.CACHE / "llm_cache.sqlite")
        self._backend = backend

    def _openai_backend(self) -> ChatBackend:
        from openai import OpenAI

        client = OpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key,
                        timeout=self.settings.timeout_s, max_retries=self.settings.max_retries)

        def call(model: str, messages: list[dict], **params: Any) -> ChatResult:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(model=model, messages=messages, **params)
            dt = time.perf_counter() - t0
            usage = resp.usage
            return ChatResult(
                content=resp.choices[0].message.content or "",
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=dt, cached=False,
            )

        return call

    def chat(self, messages: list[dict], json_mode: bool = True, max_tokens: int = 400,
             model: str | None = None) -> ChatResult:
        model = model or self.settings.model
        params: dict[str, Any] = {"temperature": self.settings.temperature, "max_tokens": max_tokens}
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        k = _key("chat", model, messages, params)
        hit = self.cache.get(k)
        if hit is not None:
            d = json.loads(hit)
            return ChatResult(**{**d, "cached": True})
        backend = self._backend or self._openai_backend()
        if self._backend is None:
            self._backend = backend
        res = backend(model, messages, **params)
        self.cache.set(k, json.dumps({**res.__dict__, "cached": False}))
        return res


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict | None:
    """Parse a JSON object from model output; tolerate code fences and leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = _JSON_OBJ.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class EmbeddingClient:
    """OpenAI-compatible embeddings with a per-text disk cache."""

    def __init__(self, settings: config.EmbedSettings | None = None, cache_path: Path | None = None,
                 backend=None):
        self.settings = settings or config.EmbedSettings()
        self.cache = _Cache(cache_path or config.CACHE / "embed_cache.sqlite")
        self._backend = backend

    def _call(self, texts: list[str]) -> list[list[float]]:
        if self._backend is not None:
            return self._backend(texts)
        from openai import OpenAI

        client = OpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key, timeout=300)
        resp = client.embeddings.create(model=self.settings.model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def embed(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Return L2-normalised embeddings (rows) for ``texts``."""
        prepared = [(prefix + t)[: self.settings.max_chars] for t in texts]
        keys = [_key("emb", self.settings.model, t) for t in prepared]
        out: list[list[float] | None] = [None] * len(texts)
        todo = []
        for i, k in enumerate(keys):
            hit = self.cache.get(k)
            if hit is not None:
                out[i] = json.loads(hit)
            else:
                todo.append(i)
        bs = self.settings.batch_size
        for s in range(0, len(todo), bs):
            idx = todo[s:s + bs]
            vecs = self._call([prepared[i] for i in idx])
            for i, v in zip(idx, vecs):
                out[i] = v
                self.cache.set(keys[i], json.dumps(v))
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)
