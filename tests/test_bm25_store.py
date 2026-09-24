from app.indexing.bm25_store import BM25Store, tokenize
from app.schemas import Chunk


def _chunk(cid, text):
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], content=text, raw_content=text,
        chunk_index=int(cid.split("#")[1]), chunk_type="markdown", title="T",
        section_path="T", source="x.md", doc_version="v1", valid_from=None,
        valid_until=None, acl="public", content_hash="h",
    )


def test_jieba_tokenizes_chinese_and_keeps_codes():
    toks = tokenize("专业版 SKU-PRO-01 的年费是 19800 元")
    assert "专业版" in toks
    assert "19800" in toks
    assert "sku-pro-01" in toks
    assert "" not in toks and "的" not in toks


async def test_build_search_persist_filter(tmp_path):
    store = BM25Store(str(tmp_path / "chunks.jsonl"))
    store.build([
        _chunk("p#0", "专业版年费 19800 元，包含销售自动化模块"),
        _chunk("p#1", "企业版支持私有化部署与定制开发"),
        _chunk("p#2", "退款政策七个工作日内无理由退款"),
    ])
    hits = store.search("专业版一年多少钱", topn=2)
    assert hits[0][0] == "p#0"
    assert all(cid != "p#2" for cid, _ in hits)

    again = BM25Store(str(tmp_path / "chunks.jsonl"))
    assert again.load() is True
    assert again.search("私有化部署", topn=1)[0][0] == "p#1"

    filtered = store.search("专业版", topn=5, allowed_ids={"p#2"})
    assert filtered == []
