from app.retrieval.reranker import BGEReranker
from app.schemas import Chunk


def _chunk(cid):
    return Chunk(
        chunk_id=cid, doc_id="d", content=cid, raw_content=cid, chunk_index=0,
        chunk_type="markdown", title="T", section_path="T", source="x.md",
        doc_version="v1", valid_from=None, valid_until=None, acl="public",
        content_hash="h",
    )


def test_rerank_returns_none_when_backend_unavailable():
    r = BGEReranker("definitely-not-a-real-model-xyz")
    assert r.rerank("q", [_chunk("d#0")], topk=1) is None
    assert r.available() is False
