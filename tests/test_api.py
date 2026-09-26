"""API 测试：注入 fake 组件，不依赖真实索引 / Ollama。"""
import json
import re

import pytest
from fastapi.testclient import TestClient

from app.api.server import create_app
from app.config import Settings
from app.schemas import Chunk


def _chunk(cid, text):
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], content=text, raw_content=text,
        chunk_index=int(cid.split("#")[1]), chunk_type="markdown", title="报价政策",
        section_path="报价", source="pricing.md", doc_version="v2",
        valid_from="2026-01-01", valid_until=None, acl="public", content_hash="h",
    )


CHUNKS = [_chunk("p#0", "专业版年费 19800 元"), _chunk("p#1", "旗舰版年费 39800 元")]


class FakeEmbedder:
    async def embed_query(self, q):
        return [1.0, 0.0]


class FakeVector:
    async def query(self, emb, topn, where=None):
        return [("p#1", 0.9), ("p#0", 0.8)]


class FakeBM25:
    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}

    def search(self, q, topn, allowed=None):  # Retriever 经 asyncio.to_thread 调用
        if allowed is not None and "p#0" not in allowed:
            return []
        return [("p#0", 1.0)]


class FakeLLM:
    def __init__(self, replies=None):
        self.replies = replies or ["年费 19800 元 [1]。"]
        self.calls: list[list[dict]] = []

    async def chat_json(self, s, u, temperature=0.1):
        return {"paraphrases": [], "hyde": None}

    async def stream(self, messages, temperature=0.3):
        self.calls.append(messages)
        yield {"type": "delta", "text": self.replies[len(self.calls) - 1]}
        yield {"type": "done", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class FakeReranker:
    status = "lazy"

    def available(self):
        return False

    def rerank(self, q, chunks, topk):
        return None


def _components(tmp_path):
    from app.indexing.manifest import Manifest

    s = Settings(corpus_dir=str(tmp_path / "corpus"), index_dir=str(tmp_path / "index"))
    mf = Manifest(str(tmp_path / "index" / "manifest.db"))
    return (s, FakeEmbedder(), FakeVector(), FakeBM25(CHUNKS),
            mf, FakeLLM(), FakeReranker())


@pytest.fixture
def client(monkeypatch, tmp_path):
    async def default_down(_base):  # 默认探活失败；alive/dead fixture 按需覆盖
        return False
    monkeypatch.setattr("app.api.routes._ollama_alive", default_down)
    app = create_app(_components(tmp_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def alive(monkeypatch):
    async def ok(_base):
        return True
    monkeypatch.setattr("app.api.routes._ollama_alive", ok)


@pytest.fixture
def dead(monkeypatch):
    async def down(_base):
        return False
    monkeypatch.setattr("app.api.routes._ollama_alive", down)


def test_health_ok(client, alive):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama"] == "ok" and body["index"] == "ready" and body["chunks"] == 2
    assert body["reranker"] == "lazy"


def test_health_degraded_without_ollama(client, dead):
    assert client.get("/health").json()["status"] == "degraded"


def test_retrieve_returns_scores_and_trace(client):
    resp = client.post("/api/retrieve", json={"question": "专业版年费"})
    assert resp.status_code == 200
    body = resp.json()
    # RRF：p#0 双路命中（向量#2 + BM25#1）得分高于 p#1（仅向量#1）
    assert [it["chunk_id"] for it in body["items"]] == ["p#0", "p#1"]
    first = body["items"][0]
    assert first["title"] == "报价政策" and "19800" in first["content"]
    assert body["rerank_available"] is False
    assert "timings_ms" in body["trace"] and "fused" in body["trace"]


def test_retrieve_validates_request(client):
    assert client.post("/api/retrieve", json={"question": ""}).status_code == 422
    assert client.post("/api/retrieve", json={"question": "x", "mode": "bad"}).status_code == 422
    assert client.post("/api/retrieve", json={"question": "x", "topk": 99}).status_code == 422


# ── /api/chat/stream（SSE）──────────────────────────────
def _events(text: str) -> list[tuple[str, dict]]:
    """解析 SSE 帧为 (event, data) 列表，跳过 ping 注释帧。"""
    out = []
    for block in re.split(r"\r?\n\r?\n", text):
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if ev is not None:
            out.append((ev, data))
    return out


def _post_chat(client, replies=None, **overrides):
    if replies is not None:
        client.app.state.generator._llm.replies = replies
    payload = {"question": "专业版年费多少", "mode": "hybrid", "use_rerank": False}
    payload.update(overrides)
    resp = client.post("/api/chat/stream", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    return _events(resp.text)


def test_chat_stream_happy_path(client):
    events = _post_chat(client)
    assert [e for e, _ in events] == [
        "status", "retrieved", "answer_delta", "done",
    ]
    assert events[0][1] == {"stage": "retrieving"}
    assert [it["chunk_id"] for it in events[1][1]["items"]] == ["p#0", "p#1"]
    done = events[-1][1]
    assert done["answer"] == "年费 19800 元 [1]。"
    assert done["citations"] == ["p#0"]
    assert done["citation_ok"] and not done["refused"]
    assert done["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert done["latency_ms"] > 0
    assert "timings_ms" in done["trace"] and done["trace"]["fused"]


def test_chat_stream_retry_sequence(client):
    events = _post_chat(client, replies=["编号 [9] 越界", "修正 [1]"])
    assert [e for e, _ in events] == [
        "status", "retrieved", "answer_delta",
        "status", "answer_delta", "done",
    ]
    assert events[3][1]["stage"] == "citation_retry"
    assert "9" in events[3][1]["detail"]
    assert events[-1][1]["answer"] == "修正 [1]"
    assert events[-1][1]["citation_ok"]


def test_chat_stream_refusal(client):
    from app.generation.prompt import REFUSAL
    events = _post_chat(client, replies=[REFUSAL])
    done = events[-1][1]
    assert done["refused"] and done["citations"] == [] and done["citation_ok"]
    assert "answer_delta" in [e for e, _ in events]


def test_chat_stream_ids_monotonic(client):
    resp = client.post("/api/chat/stream", json={"question": "q"})
    ids = re.findall(r"^id: (\d+)$", resp.text, re.MULTILINE)
    assert [int(i) for i in ids] == list(range(1, len(ids) + 1))


def test_acl_filters_chunks(client):
    """internal 语料对 public 身份不可见。"""
    secret_chunk = Chunk(
        chunk_id="s#0", doc_id="s", content="内部折扣底线", raw_content="内部折扣底线",
        chunk_index=0, chunk_type="markdown", title="内部政策",
        section_path="内部", source="internal.md", doc_version="v1",
        valid_from="2026-01-01", valid_until=None, acl="internal", content_hash="h",
    )
    client.app.state.bm.chunks["s#0"] = secret_chunk

    # public（默认）看不到 internal 块
    body = client.post("/api/retrieve", json={"question": "折扣底线"}).json()
    assert all(it["chunk_id"] != "s#0" for it in body["items"])

    # internal 身份：FakeVector 的 where 参数被忽略，但 Retriever 会用 allowed 二次过滤
    body = client.post("/api/retrieve", json={"question": "折扣底线", "acl": "internal"}).json()
    # p#0/p#1 的 acl=public，internal 身份下被 _allowed_ids 排除
    assert all(it["chunk_id"] not in ("p#0", "p#1") for it in body["items"])
