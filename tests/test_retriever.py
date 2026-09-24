from app.config import Settings
from app.retrieval.retriever import Retriever
from app.retrieval.rewrite import Rewriter
from app.schemas import Chunk


class FakeEmbedder:
    async def embed_query(self, q):
        return [1.0 if "专业" in q else 0.0, 0.0]


class FakeVector:
    def __init__(self, ids):
        self.ids = ids
        self.last_where = None

    async def query(self, emb, topn, where=None):
        self.last_where = where
        return [(cid, 0.1) for cid in self.ids]


class FakeLLM:
    async def chat_json(self, s, u, temperature=0.1):
        return {"paraphrases": ["专业版包年价格"], "hyde": None}


def _chunk(cid, text, acl="public", valid_until=None):
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], content=text, raw_content=text,
        chunk_index=int(cid.split("#")[1]), chunk_type="markdown", title="T",
        section_path="T", source="x.md", doc_version="v1", valid_from=None,
        valid_until=valid_until, acl=acl, content_hash="h",
    )


def _bm25(chunks, tmp_path, answers):
    from app.indexing.bm25_store import BM25Store
    s = BM25Store(str(tmp_path / "chunks.jsonl"))
    s.build(chunks)
    orig = s.search

    def fake_search(q, topn, allowed_ids=None):
        out = orig(q, topn, allowed_ids)
        return out if out else answers

    s.search = fake_search
    return s


class FakeReranker:
    def __init__(self, order):
        self.order, self.calls = order, 0

    def available(self):
        return True

    def rerank(self, q, chunks, topk):
        self.calls += 1
        present = {c.chunk_id for c in chunks}
        score = {cid: float(len(self.order) - i) for i, cid in enumerate(self.order)}
        return [(cid, score[cid]) for cid in self.order if cid in present][:topk]


def _settings():
    return Settings(rrf_k=60, topn_recall=20, topk_final=5)


async def test_hybrid_fuses_and_filters_expired_and_acl(tmp_path):
    chunks = [
        _chunk("p#0", "专业版年费价格政策", acl="public"),
        _chunk("p#1", "旧版专业版年费", acl="public", valid_until="2025-12-31"),
        _chunk("s#0", "销售内部提成", acl="sales"),
    ]
    bm = _bm25(chunks, tmp_path, answers=[("p#0", 1.0)])
    v = FakeVector(["p#0", "p#1", "s#0"])
    r = Retriever(FakeEmbedder(), v, bm, settings=_settings())
    res = await r.retrieve("专业版年费", mode="hybrid", today="2026-09-23")
    ids = [it.chunk.chunk_id for it in res.items]
    assert "p#0" in ids and "p#1" not in ids and "s#0" not in ids
    assert res.trace.bm25_ranking  # BM25 路被调用
    assert v.last_where is not None
    # where 必须是数值域（Chroma 不支持字符串 $gte）
    where_clause = v.last_where["$and"][1]["$or"]
    assert {"valid_until_ord": 0} in where_clause


async def test_vector_mode_skips_bm25(tmp_path):
    chunks = [_chunk("p#0", "专业版")]
    bm = _bm25(chunks, tmp_path, answers=[("p#0", 9.0)])
    r = Retriever(FakeEmbedder(), FakeVector(["p#0"]), bm, settings=_settings())
    res = await r.retrieve("专业版", mode="vector")
    assert res.trace.bm25_ranking == []


async def test_rerank_reorders_and_trace(tmp_path):
    chunks = [_chunk("p#0", "专业版年费"), _chunk("p#1", "退款")]
    bm = _bm25(chunks, tmp_path, answers=[("p#0", 1.0), ("p#1", 0.5)])
    fake_rk = FakeReranker(["p#1", "p#0"])
    r = Retriever(FakeEmbedder(), FakeVector(["p#0", "p#1"]), bm,
                  reranker=fake_rk, settings=_settings())
    res = await r.retrieve("退款", mode="hybrid", use_rerank=True, topk=2)
    assert next(it.chunk.chunk_id for it in res.items) == "p#1"
    assert res.items[0].rerank_score is not None
    assert res.rerank_available is True


async def test_rewrite_expands_vector_queries(tmp_path):
    chunks = [_chunk("p#0", "专业版包年订阅价格")]
    bm = _bm25(chunks, tmp_path, answers=[])
    r = Retriever(FakeEmbedder(), FakeVector(["p#0"]), bm,
                  rewriter=Rewriter(FakeLLM()), settings=_settings())
    res = await r.retrieve("多少钱", mode="vector", rewrite_mode="multi")
    # 原问题 + 1 条含“专业”的改写
    assert len(res.trace.vector_queries) >= 2
    assert res.trace.llm_calls == 1
