from app.indexing.vector_store import VectorStore, metadata_for_chroma
from app.schemas import Chunk


def _chunk(cid, doc="d", **meta):
    return Chunk(
        chunk_id=cid, doc_id=doc, content=f"c-{cid}", raw_content=f"c-{cid}",
        chunk_index=int(cid.split("#")[1]), chunk_type="markdown", title="T",
        section_path="T", source=f"{doc}.md", doc_version="v1",
        valid_from=meta.get("valid_from"), valid_until=meta.get("valid_until"),
        acl=meta.get("acl", "public"), content_hash="h",
    )


async def test_upsert_query_delete(tmp_path):
    vs = VectorStore(str(tmp_path / "chroma"))
    chunks = [_chunk("d#0"), _chunk("d#1"), _chunk("d#2")]
    vecs = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
    await vs.upsert(chunks, vecs)

    hits = await vs.query([1.0, 0.0, 0.0], topn=2)
    assert [cid for cid, _ in hits] == ["d#0", "d#1"]

    await vs.delete_doc("d")
    assert await vs.query([1.0, 0.0, 0.0], topn=2) == []


async def test_where_filter_acl_and_validity(tmp_path):
    vs = VectorStore(str(tmp_path / "chroma"))
    chunks = [
        _chunk("d#0", acl="public"),
        _chunk("d#1", acl="sales"),
        _chunk("p#0", doc="p", valid_until="2025-12-31"),
        _chunk("p#1", doc="p", valid_until=""),
    ]
    await vs.upsert(chunks, [[1.0, 0, 0]] * 4)
    where = {"$and": [
        {"acl": "public"},
        {"$or": [{"valid_until_ord": 0}, {"valid_until_ord": {"$gte": 20260923}}]},
    ]}
    hits = await vs.query([1.0, 0.0, 0.0], topn=10, where=where)
    assert {cid for cid, _ in hits} == {"d#0", "p#1"}


def test_metadata_none_becomes_empty_string():
    md = metadata_for_chroma(_chunk("d#0"))
    assert md["valid_until"] == "" and md["chunk_index"] == 0 and md["acl"] == "public"
    assert md["valid_until_ord"] == 0
    md_expired = metadata_for_chroma(_chunk("d#0", valid_until="2025-12-31"))
    assert md_expired["valid_until_ord"] == 20251231
