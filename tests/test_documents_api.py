"""/api/documents 测试：真 Manifest + 假索引组件（不依赖 Ollama / Chroma）。"""
import os

import pytest
from fastapi.testclient import TestClient

from app.api.server import create_app
from app.config import Settings
from app.indexing.manifest import Manifest

DOC = """---
doc_id: pricing_2026
title: 报价政策 2026
version: v2
valid_from: 2026-01-01
acl: public
---

# 报价

## 专业版

专业版年费 19800 元，含全部基础模块。
"""


class FakeEmbedder:
    async def embed_query(self, q):
        return [1.0, 0.0]

    async def embed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]


class FakeVector:
    def __init__(self):
        self.store: dict[str, list] = {}

    async def query(self, emb, topn, where=None):
        return []

    async def delete_doc(self, doc_id):
        self.store.pop(doc_id, None)

    async def upsert(self, chunks, embeddings):
        self.store[chunks[0].doc_id] = list(chunks)


class FakeBM25:
    def __init__(self):
        self.chunks: dict[str, object] = {}

    def search(self, q, topn, allowed=None):
        return []

    def build(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}

    def save(self):
        pass


class FakeLLM:
    async def chat_json(self, s, u, temperature=0.1):
        return {"paraphrases": [], "hyde": None}

    async def stream(self, messages, temperature=0.3):
        yield {"type": "delta", "text": "x"}
        yield {"type": "done", "usage": {}}


class FakeReranker:
    status = "lazy"

    def available(self):
        return False

    def rerank(self, q, chunks, topk):
        return None


@pytest.fixture
def client(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    s = Settings(corpus_dir=str(corpus), index_dir=str(tmp_path / "index"))
    mf = Manifest(str(tmp_path / "index" / "manifest.db"))
    components = (s, FakeEmbedder(), FakeVector(), FakeBM25(), mf, FakeLLM(), FakeReranker())
    app = create_app(components)
    with TestClient(app) as c:
        c.corpus = str(corpus)
        yield c


def _upload(client, name="pricing.md", content=DOC, **params):
    data = content.encode("utf-8") if isinstance(content, str) else content
    return client.post(
        "/api/documents",
        files={"file": (name, data, "text/markdown")},
        params=params,
    )


def test_upload_and_list_synced(client):
    resp = _upload(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "pricing_2026"
    assert body["overwritten"] is False
    assert body["report"]["added"] == 1
    assert body["report"]["total_chunks"] >= 1

    data = client.get("/api/documents").json()
    assert data["total_chunks"] == body["report"]["total_chunks"]
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert item["doc_id"] == "pricing_2026"
    assert item["title"] == "报价政策 2026"
    assert item["status"] == "synced"
    assert item["chunk_count"] == body["report"]["total_chunks"]
    assert item["indexed_at"] is not None
    assert os.path.exists(os.path.join(client.corpus, "pricing.md"))


def test_upload_conflict_then_overwrite(client):
    assert _upload(client).status_code == 200
    updated = DOC.replace("19800", "29800")

    resp = _upload(client, content=updated)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["conflict"] == "exists"
    assert detail["doc_id"] == "pricing_2026"
    assert detail["chunk_count"] >= 1

    resp = _upload(client, content=updated, overwrite=True)
    assert resp.status_code == 200
    body = resp.json()
    assert body["overwritten"] is True
    assert body["report"]["rebuilt"] == 1
    with open(os.path.join(client.corpus, "pricing.md"), encoding="utf-8") as f:
        assert "29800" in f.read()


def test_upload_rejects_doc_id_taken(client):
    assert _upload(client).status_code == 200
    resp = _upload(client, name="another.md", content=DOC)
    assert resp.status_code == 400
    assert "pricing_2026" in resp.json()["detail"]


def test_upload_rejects_bad_name_and_encoding(client):
    assert _upload(client, name="x.txt").status_code == 422
    assert _upload(client, name=".hidden.md").status_code == 422
    assert _upload(client, name=".md").status_code == 422
    assert _upload(client, content=b"\xff\xfe\x00bad").status_code == 422


def test_upload_rejects_bad_yaml(client):
    resp = _upload(client, content="---\ndoc_id: [unclosed\n---\n\nbody")
    assert resp.status_code == 422


def test_upload_without_frontmatter_uses_filename(client):
    resp = _upload(client, name="notes.md", content="# 随手记\n\n内容")
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == "notes"


def test_delete_document(client):
    _upload(client)
    resp = client.delete("/api/documents/pricing_2026")
    assert resp.status_code == 200
    body = resp.json()
    assert body["removed"] == "pricing_2026"
    assert body["report"]["removed"] == 1
    assert client.get("/api/documents").json()["items"] == []
    assert not os.path.exists(os.path.join(client.corpus, "pricing.md"))


def test_delete_unknown_returns_404(client):
    assert client.delete("/api/documents/nope").status_code == 404


def test_reindex_picks_up_changes(client):
    _upload(client)
    path = os.path.join(client.corpus, "pricing.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(DOC.replace("19800", "29800"))

    assert client.get("/api/documents").json()["items"][0]["status"] == "stale"
    report = client.post("/api/documents/reindex").json()
    assert report["rebuilt"] == 1
    assert client.get("/api/documents").json()["items"][0]["status"] == "synced"


def test_missing_status_and_cleanup(client):
    _upload(client)
    os.remove(os.path.join(client.corpus, "pricing.md"))
    assert client.get("/api/documents").json()["items"][0]["status"] == "missing"

    report = client.post("/api/documents/reindex").json()
    assert report["removed"] == 1
    assert client.get("/api/documents").json()["items"] == []
