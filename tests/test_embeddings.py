import json

import httpx
import pytest

from app.indexing.embeddings import EmbeddingError, OllamaEmbedder


def make_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")


async def test_embed_documents_batches_and_asserts_dim():
    seen_inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_inputs.append(payload["input"])
        return httpx.Response(200, json={"embeddings": [[0.1] * 4 for _ in payload["input"]]})

    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, batch=2, client=make_client(handler))
    out = await emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 4 for v in out)
    assert seen_inputs == [["a", "b"], ["c"]]


async def test_embed_query_returns_vector():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.2] * 4]})

    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, client=make_client(handler))
    assert await emb.embed_query("q") == [0.2] * 4


async def test_wrong_dimension_raises():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.0, 0.0]]})

    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, client=make_client(handler))
    with pytest.raises(EmbeddingError):
        await emb.embed_query("q")


async def test_retry_then_succeed(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"embeddings": [[1.0] * 4]})

    async def _instant(_):
        return 0

    monkeypatch.setattr("asyncio.sleep", _instant)
    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, client=make_client(handler))
    assert await emb.embed_query("q") == [1.0] * 4
    assert calls["n"] == 2
