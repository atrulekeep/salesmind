# P2 评测基准实施计划

> 日期：2026-09-24
> 对应 spec 第 8 节（评测体系）
> 目标：100 题评测集 + 四曲线（A/B/C/D）对比报告，数字以实测为准

## 前置状态（P0–P1 已完成）
- 4 篇种子语料 / 20 块已建索引；Ollama（bge-large、qwen3.5:9b）运行中
- 检索管线 Retriever 支持 mode/use_rerank/rewrite_mode 开关 + trace
- 40 单测全绿

## 任务分解

### Task 1：扩语料 + 重建索引 + 导出 chunk 清单
- 新增 6 篇语料（共 10 篇）：competitor_comparison / contract_sla / implementation_guide /
  security_compliance / partner_channel / case_studies
- 刻意埋入：跨文档引用、同义表述、低频实体（SKU/百分比/工作日）、过期内容、unanswerable 话题
- 全量重建索引，导出 `eval/chunk_map.json`（chunk_id → 标题/来源/正文摘要）供 Task 2 反向出题

### Task 2：100 题 eval/cases.jsonl
- 分布：fact30 / entity20 / paraphrase20 / multihop15 / temporal10 / unanswerable5
- 每题字段：id / category / question / relevant_chunk_ids / answer_keywords / should_refuse
- 题目由指定 gold chunk 反向出题，gold 精确到真实 chunk_id

### Task 3：eval/metrics.py（TDD 纯函数，不依赖 LLM）
- 检索层：Recall@k、MRR、nDCG@k（二值相关）
- 生成层：引用精确率/召回率、关键词命中率、拒答正确率
- 成本层：平均延迟、LLM 调用数、tokens
- 整体 + 六类别分组
- tests/test_metrics.py 先行

### Task 4：eval/run_eval.py 四曲线 + 报告
- asyncio + Semaphore 并发，同批 cases 依次跑 A→D
  - A: vector + 无改写 + 无rerank
  - B: hybrid(RRF) + 无改写 + 无rerank
  - C: hybrid + rerank
  - D: hybrid + rerank + both(multi+HyDE)
- 产物：`eval/reports/<ts>.json`（100×4 明细）+ `latest.md`（对比表+成本表）

### Task 5：真机跑 + 报告
- 真机执行 run_eval.py，确认四曲线数字
- reranker 模型若未下载 → C/D 曲线标记 skipped（不静默冒充）
- 生成 latest.md，README 同步真实数字

## 约束
- 诚实原则：报告数字一律实测，不背旧宣传
- corpus v1 ↔ eval v1 版本锁定
- 指标纯函数确定性，LLM-judge 仅抽样参考不进门禁
