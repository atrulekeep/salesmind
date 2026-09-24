"""app/generation/citations.py 纯函数测试：编号提取/越界校验/chunk_id 映射。"""
from app.generation.citations import (
    citation_problem,
    extract_nums,
    map_citations,
)
from app.schemas import Chunk, RetrievedChunk


def _chunk(cid):
    return Chunk(
        chunk_id=cid, doc_id="d", content="x", raw_content="x",
        chunk_index=0, chunk_type="markdown", title="报价政策", section_path="报价",
        source="pricing.md", doc_version="v2", valid_from="2026-01-01",
        valid_until=None, acl="public", content_hash="h",
    )


def _items(*cids):
    return [RetrievedChunk(chunk=_chunk(c), final_rank=i + 1)
            for i, c in enumerate(cids)]


# ── extract_nums ────────────────────────────────────────
def test_extract_basic():
    assert extract_nums("年费 19800 元 [1]，旗舰版 [2]。") == [1, 2]


def test_extract_repeated_and_adjacent():
    assert extract_nums("[1][1] 见 [3]") == [1, 1, 3]


def test_extract_ignores_non_citation_brackets():
    # 非纯数字方括号不算引用
    assert extract_nums("参见 [附录A] 和 [ 2 ]，但 [10] 算") == [10]


def test_extract_empty():
    assert extract_nums("没有任何引用。") == []


# ── citation_problem ────────────────────────────────────
def test_valid_passes():
    assert citation_problem("结论 [1] 与 [3]。", 5) is None


def test_missing_citation_fails():
    problem = citation_problem("结论没有任何标注。", 5)
    assert problem is not None and "没有" in problem


def test_out_of_range_fails():
    problem = citation_problem("结论 [1] [6] [0]。", 5)
    assert problem is not None
    assert "6" in problem and "0" in problem


def test_boundary_numbers_valid():
    assert citation_problem("[1] 和 [5]", 5) is None


# ── map_citations ───────────────────────────────────────
def test_map_to_chunk_ids_in_order():
    items = _items("a#0", "b#1", "c#2")
    assert map_citations("先说 [2]，再说 [1]。", items) == ["b#1", "a#0"]


def test_map_dedupes_keeps_first_order():
    items = _items("a#0", "b#1")
    assert map_citations("[1] [2] [1]", items) == ["a#0", "b#1"]


def test_map_skips_illegal_numbers():
    items = _items("a#0")
    assert map_citations("[1] 和 [9]", items) == ["a#0"]


def test_map_empty_answer():
    assert map_citations("无引用", _items("a#0")) == []
