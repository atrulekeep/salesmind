"""eval/metrics.py 纯函数测试：不依赖 LLM / 检索器。"""
import math

from eval.metrics import (
    CaseResult,
    aggregate,
    citation_precision,
    citation_recall,
    keyword_hit,
    mrr,
    ndcg_at_k,
    recall_at_k,
    refusal_correct,
)

REFUSAL = "根据现有资料未找到相关信息。"


# ── recall@k ───────────────────────────────────────────
def test_recall_all_hit():
    assert recall_at_k(["a", "b", "c", "d", "e"], ["a", "c"], k=5) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "b"], ["a", "x"], k=5) == 0.5


def test_recall_none():
    assert recall_at_k(["b", "c"], ["a"], k=5) == 0.0


def test_recall_respects_k():
    # gold 在第 6 位，k=5 不应命中
    retrieved = ["x", "x", "x", "x", "x", "a"]
    assert recall_at_k(retrieved, ["a"], k=5) == 0.0
    assert recall_at_k(retrieved, ["a"], k=6) == 1.0


def test_recall_empty_gold_is_zero():
    assert recall_at_k(["a"], [], k=5) == 0.0


# ── MRR ─────────────────────────────────────────────────
def test_mrr_first_position():
    assert mrr(["a", "b"], ["a"]) == 1.0


def test_mrr_second_position():
    assert mrr(["b", "a", "c"], ["a"]) == 0.5


def test_mrr_no_hit():
    assert mrr(["b", "c"], ["a"]) == 0.0


# ── nDCG@k（二值相关）──────────────────────────────────
def test_ndcg_perfect():
    # gold 排第一 → nDCG=1.0
    assert ndcg_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_ndcg_second():
    # gold 排第二
    got = ndcg_at_k(["b", "a", "c"], ["a"], k=3)
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_ndcg_two_gold():
    retrieved = ["a", "b", "c", "d", "e"]
    gold = ["a", "c"]
    dcg = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    got = ndcg_at_k(retrieved, gold, k=5)
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_ndcg_no_hit():
    assert ndcg_at_k(["b", "c"], ["a"], k=5) == 0.0


# ── 关键词命中 ──────────────────────────────────────────
def test_keyword_hit_all_present():
    assert keyword_hit("专业版年费 19800 元", ["19800"]) is True


def test_keyword_hit_missing_one():
    assert keyword_hit("专业版年费 元", ["19800", "专业版"]) is False


def test_keyword_hit_empty_keywords():
    assert keyword_hit("任意文本", []) is True


# ── 引用精确率 / 召回率 ─────────────────────────────────
def test_citation_precision_full():
    assert citation_precision(["a", "b"], ["a", "b"]) == 1.0


def test_citation_precision_partial():
    # cited 2 个，1 个命中 → 0.5
    assert citation_precision(["a", "x"], ["a", "b"]) == 0.5


def test_citation_precision_empty_cited():
    assert citation_precision([], ["a"]) == 0.0


def test_citation_recall_full():
    assert citation_recall(["a", "b"], ["a", "b"]) == 1.0


def test_citation_recall_partial():
    # gold 2 个，cited 命中 1 个 → 0.5
    assert citation_recall(["a"], ["a", "b"]) == 0.5


def test_citation_recall_empty_gold():
    assert citation_recall(["a"], []) == 0.0


# ── 拒答正确率 ──────────────────────────────────────────
def test_refusal_should_refuse_and_refused():
    assert refusal_correct(REFUSAL, should_refuse=True) is True


def test_refusal_should_refuse_but_answered():
    assert refusal_correct("年费 19800 元", should_refuse=True) is False


def test_refusal_should_answer_and_answered():
    assert refusal_correct("年费 19800 元", should_refuse=False) is True


def test_refusal_should_answer_but_refused():
    assert refusal_correct(REFUSAL, should_refuse=False) is False


# ── 聚合：整体 + 类别 + unanswerable 处理 ───────────────
def _mk(case_id, category, retrieved, gold, answer="", cited=None,
        keywords=None, should_refuse=False, latency=100.0, llm_calls=1,
        ptok=50, ctok=20):
    return CaseResult(
        case_id=case_id, category=category, retrieved_ids=retrieved,
        gold_ids=gold, answer=answer, cited_ids=cited or [],
        keywords=keywords or [], should_refuse=should_refuse,
        latency_ms=latency, llm_calls=llm_calls,
        prompt_tokens=ptok, completion_tokens=ctok,
    )


def test_aggregate_overall_retrieval_excludes_unanswerable():
    # 一个 fact（命中）+ 一个 unanswerable（gold 空）
    results = [
        _mk("c1", "fact", ["a", "b"], ["a"], answer="19800", keywords=["19800"]),
        _mk("c2", "unanswerable", ["a"], [], answer=REFUSAL, should_refuse=True),
    ]
    agg = aggregate(results, k=5)
    # 检索指标只在 fact 上算（1/1）
    assert agg["retrieval"]["recall@5"] == 1.0
    assert agg["retrieval"]["mrr"] == 1.0
    assert agg["retrieval"]["n_answerable"] == 1


def test_aggregate_by_category():
    results = [
        _mk("c1", "fact", ["a", "b"], ["a"], answer="19800", keywords=["19800"]),
        _mk("c2", "fact", ["x"], ["a"], answer="不知道", keywords=["19800"]),
        _mk("c3", "entity", ["a"], ["a"], answer="9800", keywords=["9800"]),
    ]
    agg = aggregate(results, k=5)
    assert agg["by_category"]["fact"]["retrieval"]["recall@5"] == 0.5
    assert agg["by_category"]["entity"]["retrieval"]["recall@5"] == 1.0


def test_aggregate_refusal_accuracy():
    results = [
        _mk("c1", "fact", ["a"], ["a"], answer="19800", keywords=["19800"]),
        _mk("c2", "unanswerable", ["a"], [], answer=REFUSAL, should_refuse=True),
        _mk("c3", "unanswerable", ["a"], [], answer="某值", should_refuse=True),  # 误答
    ]
    agg = aggregate(results, k=5)
    # 3 条里 2 条拒答正确 → 2/3
    assert math.isclose(agg["generation"]["refusal_accuracy"], 2 / 3, rel_tol=1e-9)


def test_aggregate_keyword_hit_excludes_refused_and_unanswerable():
    results = [
        _mk("c1", "fact", ["a"], ["a"], answer="19800", keywords=["19800"]),        # 命中
        _mk("c2", "fact", ["a"], ["a"], answer=REFUSAL, keywords=["19800"]),         # 误拒答，不计入关键词分母
        _mk("c3", "unanswerable", ["a"], [], answer=REFUSAL, should_refuse=True),    # unanswerable 不计入
    ]
    agg = aggregate(results, k=5)
    # 只有 c1 进入关键词分母 → 1/1 = 1.0
    assert agg["generation"]["keyword_hit"] == 1.0
    assert agg["generation"]["keyword_n"] == 1


def test_aggregate_cost_averages():
    results = [
        _mk("c1", "fact", ["a"], ["a"], latency=100, llm_calls=1, ptok=50, ctok=20),
        _mk("c2", "fact", ["a"], ["a"], latency=200, llm_calls=2, ptok=70, ctok=40),
    ]
    agg = aggregate(results, k=5)
    assert agg["cost"]["avg_latency_ms"] == 150.0
    assert agg["cost"]["avg_llm_calls"] == 1.5
    assert agg["cost"]["avg_prompt_tokens"] == 60.0
    assert agg["cost"]["avg_completion_tokens"] == 30.0


# ── 失败用例（error 非空）剔除 ─────────────────────────
def test_aggregate_excludes_error_cases():
    ok = _mk("c1", "fact", ["a"], ["a"], answer="19800", keywords=["19800"],
             latency=100)
    bad = _mk("c2", "fact", ["x"], ["a"], answer="", latency=999)
    bad.error = "ReadTimeout: boom"
    agg = aggregate([ok, bad], k=5)
    assert agg["n"] == 1
    assert agg["n_errors"] == 1
    assert agg["retrieval"]["recall@5"] == 1.0
    assert agg["cost"]["avg_latency_ms"] == 100.0
    assert set(agg["by_category"]) == {"fact"}
    assert agg["by_category"]["fact"]["cost"]["n"] == 1


def test_aggregate_all_errors_returns_zeros():
    bad = _mk("c1", "fact", [], ["a"])
    bad.error = "boom"
    agg = aggregate([bad], k=5)
    assert agg["n"] == 0
    assert agg["n_errors"] == 1
    assert agg["retrieval"]["recall@5"] == 0.0
    assert agg["generation"]["refusal_accuracy"] == 0.0
    assert agg["by_category"] == {}
