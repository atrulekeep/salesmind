"""评测指标纯函数：不依赖 LLM / 检索器，确定性、可进 CI。

分为三层（对齐 spec §8.2）：
- 检索层：Recall@k、MRR、nDCG@k（二值相关）
- 生成层：引用精确率/召回率、关键词命中率、拒答正确率
- 成本层：平均延迟、LLM 调用数、prompt/completion tokens

聚合时：检索/引用/关键词指标只在 answerable（gold 非空）子集上计算，
拒答正确率在全集上计算，成本在全集上平均。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from app.generation.prompt import REFUSAL as REFUSAL_PHRASE

# 检索层默认 k
DEFAULT_K = 5


# ──────────────────────────────────────────────────────
# 检索层
# ──────────────────────────────────────────────────────
def recall_at_k(retrieved: list[str], gold: list[str], k: int = DEFAULT_K) -> float:
    """前 k 个命中 gold 的比例；gold 为空 → 0.0。"""
    if not gold:
        return 0.0
    topk = retrieved[:k]
    hit = sum(1 for cid in gold if cid in topk)
    return hit / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    """第一个 gold 出现位置的倒数；无命中 → 0.0。"""
    gold_set = set(gold)
    for i, cid in enumerate(retrieved):
        if cid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def _dcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    s = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in gold:
            s += 1.0 / math.log2(i + 2)  # i+2 因 log2(1)=0
    return s


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int = DEFAULT_K) -> float:
    """二值相关 nDCG@k；gold 为空 → 0.0。"""
    if not gold:
        return 0.0
    gold_set = set(gold)
    dcg = _dcg_at_k(retrieved, gold_set, k)
    # 理想排序：所有 gold 排最前
    ideal = list(gold) + [c for c in retrieved if c not in gold_set]
    idcg = _dcg_at_k(ideal, gold_set, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ──────────────────────────────────────────────────────
# 生成层
# ──────────────────────────────────────────────────────
def keyword_hit(answer: str, keywords: list[str]) -> bool:
    """所有必需关键词都出现才算命中（substring）。"""
    return all(kw in answer for kw in keywords)


def citation_precision(cited: list[str], gold: list[str]) -> float:
    """|cited ∩ gold| / |cited|；cited 为空 → 0.0。"""
    if not cited:
        return 0.0
    hit = sum(1 for cid in cited if cid in set(gold))
    return hit / len(cited)


def citation_recall(cited: list[str], gold: list[str]) -> float:
    """|cited ∩ gold| / |gold|；gold 为空 → 0.0。"""
    if not gold:
        return 0.0
    hit = sum(1 for cid in cited if cid in set(gold))
    return hit / len(gold)


def refusal_correct(answer: str, should_refuse: bool,
                    phrase: str = REFUSAL_PHRASE) -> bool:
    """拒答正确：是否拒答 == 是否应拒答。"""
    refused = phrase in answer
    return refused == should_refuse


# ──────────────────────────────────────────────────────
# 聚合
# ──────────────────────────────────────────────────────
@dataclass
class CaseResult:
    case_id: str
    category: str
    retrieved_ids: list[str]
    gold_ids: list[str]
    answer: str
    cited_ids: list[str]
    keywords: list[str]
    should_refuse: bool
    latency_ms: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None  # 非 None 表示该用例执行失败，聚合时剔除


def _safe_mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _retrieval_block(results: list[CaseResult], k: int) -> dict:
    rec = [recall_at_k(r.retrieved_ids, r.gold_ids, k) for r in results]
    mrrs = [mrr(r.retrieved_ids, r.gold_ids) for r in results]
    ndcgs = [ndcg_at_k(r.retrieved_ids, r.gold_ids, k) for r in results]
    return {
        "recall@5": _safe_mean(rec),
        "mrr": _safe_mean(mrrs),
        "ndcg@5": _safe_mean(ndcgs),
        "n_answerable": len(results),
    }


def _generation_block(results: list[CaseResult]) -> dict:
    answerable = [r for r in results if r.gold_ids]  # 非 unanswerable
    # 关键词：只在“应答且未拒答”的 case 上算（误拒答不计入分母）
    kw_cases = [r for r in answerable if REFUSAL_PHRASE not in r.answer]
    kw_hits = [keyword_hit(r.answer, r.keywords) for r in kw_cases]
    # 引用：在 answerable 上算
    cit_p = [citation_precision(r.cited_ids, r.gold_ids) for r in answerable]
    cit_r = [citation_recall(r.cited_ids, r.gold_ids) for r in answerable]
    # 拒答正确率：全集
    ref_correct = [refusal_correct(r.answer, r.should_refuse) for r in results]
    return {
        "keyword_hit": _safe_mean([1.0 if h else 0.0 for h in kw_hits]),
        "keyword_n": len(kw_cases),
        "citation_precision": _safe_mean(cit_p),
        "citation_recall": _safe_mean(cit_r),
        "refusal_accuracy": _safe_mean([1.0 if c else 0.0 for c in ref_correct]),
        "refusal_n": len(results),
    }


def _cost_block(results: list[CaseResult]) -> dict:
    return {
        "avg_latency_ms": _safe_mean([r.latency_ms for r in results]),
        "avg_llm_calls": _safe_mean([float(r.llm_calls) for r in results]),
        "avg_prompt_tokens": _safe_mean([float(r.prompt_tokens) for r in results]),
        "avg_completion_tokens": _safe_mean([float(r.completion_tokens) for r in results]),
        "n": len(results),
    }


def aggregate(results: list[CaseResult], k: int = DEFAULT_K) -> dict:
    """整体 + 六类别聚合。失败用例（error 非空）不计入任何分母。"""
    n_errors = sum(1 for r in results if r.error)
    results = [r for r in results if not r.error]
    answerable = [r for r in results if r.gold_ids]
    cats = sorted({r.category for r in results})
    by_cat = {}
    for cat in cats:
        sub = [r for r in results if r.category == cat]
        sub_answerable = [r for r in sub if r.gold_ids]
        by_cat[cat] = {
            "retrieval": _retrieval_block(sub_answerable, k),
            "generation": _generation_block(sub),
            "cost": _cost_block(sub),
        }
    return {
        "retrieval": _retrieval_block(answerable, k),
        "generation": _generation_block(results),
        "cost": _cost_block(results),
        "by_category": by_cat,
        "n": len(results),
        "n_errors": n_errors,
    }
