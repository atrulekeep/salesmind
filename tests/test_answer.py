"""app/generation/answer.py 测试：mock LLM 覆盖校验通过/自纠成功/自纠失败/拒答。"""
from app.generation.answer import AnswerGenerator
from app.generation.prompt import REFUSAL
from app.schemas import Chunk, RetrievedChunk


def _chunk(cid):
    return Chunk(
        chunk_id=cid, doc_id="d", content="x", raw_content="x",
        chunk_index=0, chunk_type="markdown", title="报价政策", section_path="报价",
        source="pricing.md", doc_version="v2", valid_from="2026-01-01",
        valid_until=None, acl="public", content_hash="h",
    )


ITEMS = [RetrievedChunk(chunk=_chunk("a#0"), final_rank=1),
         RetrievedChunk(chunk=_chunk("b#1"), final_rank=2)]


class FakeLLM:
    """按调用次序回放预设回复；记录收到的 messages 供断言。"""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls: list[list[dict]] = []

    async def stream(self, messages, temperature=0.3):
        self.calls.append(messages)
        yield {"type": "delta", "text": self.replies[len(self.calls) - 1]}
        yield {"type": "done",
               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


async def _collect(gen):
    return [ev async for ev in gen]


async def test_first_pass_valid():
    llm = FakeLLM(["年费 19800 元 [1]。"])
    events = await _collect(AnswerGenerator(llm).generate("年费", ITEMS))
    assert [e["type"] for e in events] == ["answer_delta", "done"]
    done = events[-1]
    assert done["answer"] == "年费 19800 元 [1]。"
    assert done["citations"] == ["a#0"]
    assert done["citation_ok"] and not done["refused"]
    assert done["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}


async def test_retry_success():
    llm = FakeLLM(["编号 [9] 越界的回答", "修正后 [2] 合法"])
    events = await _collect(AnswerGenerator(llm).generate("问题", ITEMS))
    types = [e["type"] for e in events]
    assert types == ["answer_delta", "status", "answer_delta", "done"]
    retry = events[1]
    assert retry["stage"] == "citation_retry" and "9" in retry["detail"]
    done = events[-1]
    assert done["answer"] == "修正后 [2] 合法"
    assert done["citations"] == ["b#1"] and done["citation_ok"]
    # usage 累计两轮
    assert done["usage"] == {"prompt_tokens": 20, "completion_tokens": 10}
    # 自纠消息带上了首轮答案与纠错指令
    second_call = llm.calls[1]
    assert second_call[2]["role"] == "assistant"
    assert second_call[3]["role"] == "user" and "引用规则" in second_call[3]["content"]


async def test_retry_exhausted_marks_false():
    llm = FakeLLM(["[9] 非法", "仍然 [9] 非法"])
    events = await _collect(AnswerGenerator(llm).generate("问题", ITEMS))
    assert len(llm.calls) == 2  # 只自纠一次，不无限重试
    done = events[-1]
    assert done["answer"] == "仍然 [9] 非法"
    assert not done["citation_ok"]
    assert done["citations"] == []  # 非法编号不映射


async def test_missing_citation_triggers_retry():
    llm = FakeLLM(["没有标注的回答", "补上 [1]"])
    events = await _collect(AnswerGenerator(llm).generate("问题", ITEMS))
    assert events[1]["type"] == "status"
    assert "没有" in events[1]["detail"]
    assert events[-1]["citation_ok"]


async def test_refusal_skips_validation():
    llm = FakeLLM([REFUSAL])
    events = await _collect(AnswerGenerator(llm).generate("工资多少", ITEMS))
    assert [e["type"] for e in events] == ["answer_delta", "done"]  # 无自纠
    done = events[-1]
    assert done["refused"] and done["citations"] == [] and done["citation_ok"]
