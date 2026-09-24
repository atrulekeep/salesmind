import pytest

from app.retrieval.fuser import rrf


def test_same_id_from_two_routes_merges():
    out = rrf([["a", "b"], ["a", "c"]], k=60)
    assert out[0][0] == "a"
    expected = 1 / 61 + 1 / 61
    assert out[0][1] == pytest.approx(expected)


def test_order_and_topn():
    out = rrf([["a", "b", "c"], ["b", "a"]], k=60, topn=2)
    assert [cid for cid, _ in out] == ["a", "b"]
    assert len(out) == 2


def test_empty_lists_and_weights():
    assert rrf([]) == []
    assert rrf([[]]) == []
    # 权重 0 的一路不贡献
    out = rrf([["a"], ["b"]], weights=[1.0, 0.0])
    assert out[0][0] == "a"
