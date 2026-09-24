"""流式生成 + 引用程序化校验 + 一次自纠（spec §7 三道防线之后两道）。

事件流（API 直接透传为 SSE）：
- answer_delta：增量文本
- status(stage="citation_retry")：首轮引用校验失败，开始自纠（前端应清空已渲染文本）
- done：{answer, citations, citation_ok, refused, usage, latency_ms}

诚实原则：自纠失败不静默——citation_ok=false 照常返回。
"""
import time
from collections.abc import AsyncIterator

from app.generation.citations import citation_problem, map_citations
from app.generation.prompt import REFUSAL, build_messages
from app.schemas import RetrievedChunk

RETRY_INSTRUCTION = """你上一次的回答违反了引用规则：{problem}
请基于同样的引用资料重新输出完整回答：
1. 每个结论后用 [编号] 标注来源，编号必须取自引用资料中给出的编号；
2. 资料不足以得出答案时，只回复：{refusal}
直接输出修正后的回答，不要任何解释或道歉。"""


class AnswerGenerator:
    def __init__(self, llm):
        self._llm = llm

    async def generate(self, question: str,
                       items: list[RetrievedChunk]) -> AsyncIterator[dict]:
        t0 = time.perf_counter()
        messages = build_messages(question, [it.chunk for it in items])
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        answer, problem = "", None
        for attempt in (1, 2):
            parts: list[str] = []
            async for ev in self._llm.stream(messages):
                if ev["type"] == "delta":
                    parts.append(ev["text"])
                    yield {"type": "answer_delta", "text": ev["text"]}
                else:
                    for k in usage:
                        usage[k] += ev["usage"].get(k, 0)
            answer = "".join(parts)
            refused = REFUSAL in answer
            # 拒答无引用属合法，豁免校验
            problem = None if refused else citation_problem(answer, len(items))
            if problem is None:
                break
            if attempt == 1:
                yield {"type": "status", "stage": "citation_retry", "detail": problem}
                messages = messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": RETRY_INSTRUCTION.format(
                        problem=problem, refusal=REFUSAL)},
                ]

        refused = REFUSAL in answer
        yield {
            "type": "done",
            "answer": answer,
            "citations": [] if refused else map_citations(answer, items),
            "citation_ok": problem is None,
            "refused": refused,
            "usage": usage,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
