from app.indexing.chunker import chunk_document, recursive_split
from app.indexing.loader import ScannedDoc


def _doc(body, doc_id="d", **meta_over):
    meta = {
        "path": "data/corpus/d.md", "doc_id": doc_id, "title": "D", "body": body,
        "md5": "x", "doc_version": "v1", "valid_from": None,
        "valid_until": None, "acl": "public",
    }
    meta.update(meta_over)
    return ScannedDoc(**meta)


def test_recursive_split_respects_size_and_overlap():
    text = "句子。" * 400  # 1200 字符
    pieces = recursive_split(text, size=512, overlap=64)
    assert len(pieces) >= 3
    assert all(len(p) <= 512 for p in pieces)
    # 相邻块存在重叠内容
    assert pieces[0][-20:] in pieces[1]


def test_sections_get_title_path_injection():
    body = "# 产品手册\n\n概述内容。\n\n## 套餐版本\n\n专业版很强。\n"
    chunks = chunk_document(_doc(body))
    assert chunks[0].section_path == "产品手册"
    assert any(c.section_path == "产品手册 > 套餐版本" for c in chunks)
    target = next(c for c in chunks if c.section_path == "产品手册 > 套餐版本")
    assert target.content.startswith("标题：产品手册 > 套餐版本\n")
    assert target.chunk_id == "d#" + str(target.chunk_index)


def test_chunk_indices_continuous_and_hash_present():
    body = "# A\n\n" + "内容。" * 300
    chunks = chunk_document(_doc(body))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.content_hash) == 32 for c in chunks)
    assert all(c.doc_id == "d" for c in chunks)
    assert all(c.chunk_type in ("markdown", "table") for c in chunks)


def test_table_kept_whole_with_header():
    table = "| SKU | 价格 |\n| --- | --- |\n| P01 | 19800 |\n| P02 | 29800 |\n"
    body = f"# 报价\n\n价格见下。\n\n{table}\n表格之后说明。\n"
    chunks = chunk_document(_doc(body))
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "P01" in table_chunks[0].raw_content and "P02" in table_chunks[0].raw_content
    assert "| SKU | 价格 |" in table_chunks[0].raw_content
