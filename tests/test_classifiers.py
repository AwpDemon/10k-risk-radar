"""Classifier tests. The LLM is always mocked: CI never needs a model server."""
import json

import numpy as np

from riskradar.classify import baseline, embed_lr
from riskradar.classify.llm_classifier import (FewShotExample, build_messages, classify_many,
                                               classify_one, validate)
from riskradar.llm import ChatResult, EmbeddingClient, LLMClient, parse_json
from riskradar.taxonomy import KEYS


def test_keyword_baseline_obvious_cases():
    assert baseline.predict("A cyberattack could disrupt our systems", "ransomware and hacking") == "cyber_tech"
    assert baseline.predict("We may be unable to obtain reinsurance", "reinsurers may fail to pay") == "reinsurance"
    assert baseline.predict("Nothing matches", "zzz") == "other"


def test_prefix_keywords():
    assert baseline.keyword_scores("Outsourcing to vendors", "")["operational_people"] > 0


class MockBackend:
    """Mock chat backend: answers keyed on the risk-factor heading; counts calls."""

    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def __call__(self, model, messages, **params):
        self.calls += 1
        heading = messages[-1]["content"].split("\n")[1]
        return ChatResult(self.answers.get(heading, "not json"), 100, 20, 0.5, False)


def _client(tmp_path, answers):
    backend = MockBackend(answers)
    return LLMClient(cache_path=tmp_path / "c.sqlite", backend=backend), backend


def test_llm_structured_output_and_cache(tmp_path):
    ans = {"Cyber heading": json.dumps({"primary_category": "cyber_tech", "secondary_categories": ["operational_people"],
                                        "severity": 3, "ai_related": False, "rationale": "x"})}
    client, backend = _client(tmp_path, ans)
    p = classify_one(client, "Cyber heading", "body")
    assert (p.primary, p.secondary, p.severity, p.valid) == ("cyber_tech", ["operational_people"], 3, True)
    p2 = classify_one(client, "Cyber heading", "body")
    assert p2.cached and backend.calls == 1


def test_invalid_output_is_counted_not_hidden(tmp_path):
    client, _ = _client(tmp_path, {})
    p = classify_one(client, "Unknown heading", "body")
    assert p.primary == "other" and p.valid is False


def test_validate_coerces_and_rejects():
    obj, ok = validate('```json\n{"primary_category": "Cyber Tech", "severity": 9, '
                       '"secondary_categories": ["nonsense"]}\n```')
    assert ok and obj["primary"] == "cyber_tech" and obj["severity"] is None and obj["secondary"] == []
    obj, ok = validate('{"primary_category": "weather"}')
    assert not ok and obj["primary"] == "other"
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_few_shot_messages_alternate_roles():
    shots = [FewShotExample("h1", "b1", "credit"), FewShotExample("h2", "b2", "reinsurance")]
    msgs = build_messages("H", "B", shots)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"])["primary_category"] == "credit"
    assert all(k in msgs[0]["content"] for k in KEYS)


def test_classify_many_preserves_order(tmp_path):
    ans = {f"h{i}": json.dumps({"primary_category": k}) for i, k in enumerate(KEYS)}
    client, _ = _client(tmp_path, ans)
    res = classify_many(client, [(f"h{i}", "b") for i in range(len(KEYS))], workers=3, progress=False)
    assert [r.primary for r in res] == list(KEYS)


def test_embedding_client_normalises_and_caches(tmp_path):
    calls = []

    def backend(texts):
        calls.append(len(texts))
        return [[3.0, 4.0] for _ in texts]

    e = EmbeddingClient(cache_path=tmp_path / "e.sqlite", backend=backend)
    v = e.embed(["a", "b"])
    assert np.allclose(v, [[0.6, 0.8], [0.6, 0.8]])
    e.embed(["a", "b"])
    assert calls == [2]


def test_embed_lr_cross_val_predict_learns_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 5)) + 1, rng.normal(0, 0.1, (20, 5)) - 1])
    y = np.array(["a"] * 20 + ["b"] * 20)
    assert (embed_lr.cross_val_predict(X, y) == y).mean() == 1.0
