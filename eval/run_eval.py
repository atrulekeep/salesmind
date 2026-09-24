"""四曲线（A/B/C/D）端到端评测：检索 → 生成 → 指标聚合 → 报告。

四曲线配置（spec §6.1）：
- A: vector + 无改写 + 无 rerank
- B: hybrid(RRF) + 无改写 + 无 rerank
- C: hybrid + rerank
- D: hybrid + rerank + both(multi + HyDE)

产物：
- eval/reports/<ts>.json：100 题 × 四曲线完整明细 + per-curve aggregate
- eval/reports/latest.md：四曲线对比表 + 类别表 + 成本表 + 分析

诚实原则：reranker 不可用时 C/D 降级等同 B，报告明确标注，不静默冒充。
tokens 仅含生成阶段（改写阶段未计）。

跑法：.venv/bin/python -m eval.run_eval
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from app.cli import _build_components
from app.generation.citations import map_citations
from app.generation.prompt import build_messages
from app.llm import OllamaLLM
from app.retrieval.retriever import Retriever
from app.retrieval.rewrite import Rewriter
from eval.metrics import CaseResult, aggregate

EVAL_DIR = Path(__file__).resolve().parent
CASES = EVAL_DIR / "cases.jsonl"
REPORTS = EVAL_DIR / "reports"

CURVES: dict[str, dict] = {
    "A": {"mode": "vector", "use_rerank": False, "rewrite_mode": "none",
          "label": "vector + 无改写 + 无 rerank"},
    "B": {"mode": "hybrid", "use_rerank": False, "rewrite_mode": "none",
          "label": "hybrid(RRF) + 无改写 + 无 rerank"},
    "C": {"mode": "hybrid", "use_rerank": True, "rewrite_mode": "none",
          "label": "hybrid + rerank"},
    "D": {"mode": "hybrid", "use_rerank": True, "rewrite_mode": "both",
          "label": "hybrid + rerank + both(multi+HyDE)"},
}

# 串行：Ollama 单 GPU，长思考并发会排队导致流式读超时；快题下串行也足够快
CONCURRENCY = int(os.environ.get("SALESMIND_EVAL_CONCURRENCY", "1"))
MAX_ATTEMPTS = 2  # 单用例首次失败后重试一次

# qwen3 系 thinking：评测默认关闭（单题 1.4s vs 开启 57s，且 P2 主测检索）；
# SALESMIND_EVAL_THINK=1 可按模型默认（含思考）复测
_THINK_ENV = os.environ.get("SALESMIND_EVAL_THINK", "0").strip().lower()
THINK_ENABLED = _THINK_ENV in ("1", "true", "on", "yes")


def load_cases() -> list[dict]:
    cases = []
    with CASES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


async def run_one(retriever: Retriever, llm, case: dict, cfg: dict,
                 today: str, topk: int) -> tuple[CaseResult, bool]:
    t0 = time.perf_counter()
    result = await retriever.retrieve(
        case["question"],
        mode=cfg["mode"],
        use_rerank=cfg["use_rerank"],
        rewrite_mode=cfg["rewrite_mode"],
        acl="public",
        today=today,
        topk=topk,
    )
    retrieved_ids = [it.chunk.chunk_id for it in result.items]
    messages = build_messages(case["question"], [it.chunk for it in result.items])
    parts: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    async for ev in llm.stream(messages):
        if ev["type"] == "delta":
            parts.append(ev["text"])
        else:
            usage = ev["usage"]
    answer = "".join(parts)
    cited_ids = map_citations(answer, result.items)
    latency_ms = (time.perf_counter() - t0) * 1000
    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        retrieved_ids=retrieved_ids,
        gold_ids=case["relevant_chunk_ids"],
        answer=answer,
        cited_ids=cited_ids,
        keywords=case["answer_keywords"],
        should_refuse=case["should_refuse"],
        latency_ms=latency_ms,
        llm_calls=result.trace.llm_calls + 1,  # 改写 + 生成
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    ), result.rerank_available


def _error_result(case: dict, err: Exception) -> CaseResult:
    """失败用例占位：不进入任何指标分母，错误信息留档。"""
    msg = f"{type(err).__name__}: {err}"[:300]
    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        retrieved_ids=[],
        gold_ids=case["relevant_chunk_ids"],
        answer="",
        cited_ids=[],
        keywords=case["answer_keywords"],
        should_refuse=case["should_refuse"],
        latency_ms=0.0,
        llm_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        error=msg,
    )


async def run_curve(curve_id: str, cfg: dict, cases: list[dict],
                    retriever: Retriever, llm, today: str, topk: int,
                    sem: asyncio.Semaphore) -> dict:
    results: list[CaseResult] = []
    rerank_flags: list[bool] = []
    errors: list[dict] = []
    print(f"  [{curve_id}] 开始（{cfg['label']}）", flush=True)
    for i, case in enumerate(cases, 1):
        cr: CaseResult | None = None
        rr = False
        last_err: Exception | None = None
        async with sem:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    cr, rr = await run_one(retriever, llm, case, cfg, today, topk)
                    break
                except Exception as e:  # noqa: BLE001 - 单题失败不拖垮整批
                    last_err = e
                    if attempt < MAX_ATTEMPTS:
                        await asyncio.sleep(2)
        if cr is None:
            cr = _error_result(case, last_err)  # type: ignore[arg-type]
            errors.append({"case_id": case["id"], "error": cr.error})
            print(f"  [{curve_id}] case {case['id']} 失败（已重试）：{cr.error}",
                  flush=True)
        else:
            rerank_flags.append(rr)
        results.append(cr)
        if i % 10 == 0:
            print(f"  [{curve_id}] {i}/{len(cases)} done"
                  f"（失败 {len(errors)}）", flush=True)
    agg = aggregate(results)
    rerank_available = bool(rerank_flags) and all(rerank_flags)
    return {
        "curve": curve_id,
        "label": cfg["label"],
        "config": {k: cfg[k] for k in ("mode", "use_rerank", "rewrite_mode")},
        "rerank_available": rerank_available,
        "n_errors": len(errors),
        "errors": errors,
        "aggregate": agg,
        "results": [asdict(r) for r in results],
    }


async def main_async() -> None:
    today = date.today().isoformat()  # noqa: DTZ011
    print(f"评测日期 today={today} 并发={CONCURRENCY} "
          f"thinking={'默认' if THINK_ENABLED else '关闭'}", flush=True)
    s, embedder, vs, bm, _, llm, reranker = _build_components()
    llm = OllamaLLM(
        s.ollama_base_url, s.llm_model,
        timeout=s.request_timeout, think=None if THINK_ENABLED else False,
    )
    retriever = Retriever(
        embedder, vs, bm,
        reranker=reranker, rewriter=Rewriter(llm), settings=s,
    )
    cases = load_cases()
    topk = s.topk_final
    print(f"载入 {len(cases)} 题，topk={topk}", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)

    curve_results = await asyncio.gather(
        *(run_curve(cid, cfg, cases, retriever, llm, today, topk, sem)
          for cid, cfg in CURVES.items())
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    json_path = REPORTS / f"{ts}.json"
    payload = {
        "generated_at": ts,
        "today": today,
        "topk": topk,
        "concurrency": CONCURRENCY,
        "thinking": "default" if THINK_ENABLED else "off",
        "n_cases": len(cases),
        "curves": curve_results,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render_md(payload)
    md_path = REPORTS / "latest.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\n报告 JSON：{json_path}", flush=True)
    print(f"报告 MD ：{md_path}", flush=True)


def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def render_md(payload: dict) -> str:
    curves = payload["curves"]
    lines: list[str] = []
    lines.append("# SalesMind P2 评测报告（四曲线对比）\n")
    lines.append(f"- 生成时间：{payload['generated_at']}")
    lines.append(f"- 评测日期 today：{payload['today']}")
    lines.append(f"- 题目数：{payload['n_cases']}，topk={payload['topk']}")
    lines.append(f"- 并发：{payload['concurrency']}")
    lines.append(f"- 生成 thinking：{payload.get('thinking', '?')}"
                 "（off=显式关闭推理链，只产直接答案）")
    lines.append("- tokens 仅含生成阶段（改写阶段未计入）")
    total_err = sum(c.get("n_errors", 0) for c in curves)
    lines.append(f"- 执行失败用例：{total_err}（已重试一次；失败用例不计入任何指标分母）\n")

    # ── 四曲线总表 ──
    lines.append("## 四曲线总表\n")
    lines.append("| 曲线 | 配置 | rerank | Recall@5 | MRR | nDCG@5 |"
                 " 引用精确率 | 引用召回率 | 关键词命中 | 拒答正确率 | 平均延迟ms | 失败数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in curves:
        r = c["aggregate"]["retrieval"]
        g = c["aggregate"]["generation"]
        cost = c["aggregate"]["cost"]
        rr = "✓" if c["rerank_available"] else ("—" if c["config"]["use_rerank"] else "N/A")
        lines.append(
            f"| {c['curve']} | {c['label']} | {rr} |"
            f" {_f(r['recall@5'])} | {_f(r['mrr'])} | {_f(r['ndcg@5'])} |"
            f" {_f(g['citation_precision'])} | {_f(g['citation_recall'])} |"
            f" {_f(g['keyword_hit'])} | {_f(g['refusal_accuracy'])} |"
            f" {_f(cost['avg_latency_ms'], 0)} | {c.get('n_errors', 0)} |"
        )
    lines.append("")

    # ── 失败用例清单 ──
    failed = [(c["curve"], e) for c in curves for e in c.get("errors", [])]
    if failed:
        lines.append("## 执行失败用例\n")
        for curve_id, e in failed:
            lines.append(f"- [{curve_id}] {e['case_id']}：{e['error']}")
        lines.append("")

    # ── 降级说明（区分：C 降级=等同B；D 降级=仅缺精排，改写仍生效）──
    degraded = [c["curve"] for c in curves
                if c["config"]["use_rerank"] and not c["rerank_available"]]
    if degraded:
        notes = []
        if "C" in degraded:
            notes.append("C 已降级，实际等同 B（hybrid 无 rerank）")
        if "D" in degraded:
            notes.append("D 的 rerank 已降级，实际配置为 "
                         "「hybrid + both 改写（无精排）」，其相对 B 的差异只代表改写增益")
        notes.append("下载 reranker 后重跑可获真实精排数字")
        lines.append("> ⚠️ reranker 不可用：" + "；".join(notes) + "。\n")

    # ── 类别表（六类 × 四曲线检索/生成核心指标）──
    lines.append("## 按类别（检索 Recall@5 / 关键词命中）\n")
    cats = sorted({k for c in curves for k in c["aggregate"]["by_category"]})
    lines.append("| 类别 | " + " | ".join(
        f"{c['curve']} Recall@5 / 关键词" for c in curves) + " |")
    lines.append("|---|" + "|".join(["---"] * len(curves)) + "|")
    for cat in cats:
        row = [cat]
        for c in curves:
            bc = c["aggregate"]["by_category"].get(cat, {})
            rec = bc.get("retrieval", {}).get("recall@5", 0.0)
            kw = bc.get("generation", {}).get("keyword_hit", 0.0)
            row.append(f"{_f(rec)} / {_f(kw)}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── 成本表 ──
    lines.append("## 成本（成功用例平均）\n")
    lines.append("| 曲线 | 平均延迟ms | 平均 LLM 调用 | 平均 prompt tok | 平均 completion tok |")
    lines.append("|---|---|---|---|---|")
    for c in curves:
        cost = c["aggregate"]["cost"]
        lines.append(
            f"| {c['curve']} | {_f(cost['avg_latency_ms'], 0)} |"
            f" {_f(cost['avg_llm_calls'], 2)} |"
            f" {_f(cost['avg_prompt_tokens'], 0)} |"
            f" {_f(cost['avg_completion_tokens'], 0)} |"
        )
    lines.append("")

    # ── 分析提示 ──
    lines.append("## 分析要点\n")
    a = curves[0]["aggregate"]["retrieval"]
    b = next((c for c in curves if c["curve"] == "B"), None)
    b_rec = b["aggregate"]["retrieval"]["recall@5"] if b else 0.0
    lines.append(f"- A(vector) Recall@5={_f(a['recall@5'])}，"
                 f"B(hybrid) Recall@5={_f(b_rec)}："
                 f"hybrid 较 vector {'提升' if b_rec > a['recall@5'] else '未提升'}"
                 f" {abs(b_rec - a['recall@5'])*100:.1f} 个百分点。")
    c_obj = next((x for x in curves if x["curve"] == "C"), None)
    d_obj = next((x for x in curves if x["curve"] == "D"), None)
    if c_obj and "C" not in degraded:
        c_rec = c_obj["aggregate"]["retrieval"]["recall@5"]
        lines.append(f"- C(rerank) Recall@5={_f(c_rec)}："
                     f"精排相对纯 hybrid {'提升' if c_rec > b_rec else '未提升'}"
                     f" {abs(c_rec - b_rec)*100:.1f} 个百分点。")
    if d_obj:
        d_rec = d_obj["aggregate"]["retrieval"]["recall@5"]
        d_gain = ("改写+精排" if "D" not in degraded else "改写（精排降级缺失）")
        lines.append(f"- D(both) Recall@5={_f(d_rec)}："
                     f"{d_gain}相对纯 hybrid {'提升' if d_rec > b_rec else '未提升'}"
                     f" {abs(d_rec - b_rec)*100:.1f} 个百分点。")
    kw_a = curves[0]["aggregate"]["generation"]["keyword_hit"]
    lines.append(f"- 关键词命中率（A）={_f(kw_a)}：反映生成阶段答案对关键事实的覆盖。")
    ref = curves[0]["aggregate"]["generation"]["refusal_accuracy"]
    lines.append(f"- 拒答正确率（全集）={_f(ref)}：含 5 题 unanswerable + 时效陷阱。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
