from app.llm import LLMError
from app.retrieval.rewrite import Rewriter


class FakeLLM:
    def __init__(self, payload=None, raise_exc=False):
        self.payload, self.raise_exc = payload, raise_exc
        self.systems = []

    async def chat_json(self, system, user, temperature=0.1):
        self.systems.append(system)
        if self.raise_exc:
            raise LLMError("boom")
        return self.payload


async def test_none_mode_no_call():
    llm = FakeLLM({"paraphrases": ["x"], "hyde": "y"})
    out = await Rewriter(llm).rewrite("问题", "none")
    assert out.paraphrases == [] and out.hyde is None and out.llm_calls == 0
    assert llm.systems == []


async def test_both_mode_one_call():
    llm = FakeLLM({"paraphrases": ["同义1", "同义2"], "hyde": "假想答案"})
    out = await Rewriter(llm).rewrite("问题", "both")
    assert out.paraphrases == ["同义1", "同义2"]
    assert out.hyde == "假想答案"
    assert out.llm_calls == 1 and len(llm.systems) == 1


async def test_multi_and_hyde_modes():
    out = await Rewriter(FakeLLM({"paraphrases": ["a"]})).rewrite("q", "multi")
    assert out.paraphrases == ["a"] and out.hyde is None
    out = await Rewriter(FakeLLM({"hyde": "h"})).rewrite("q", "hyde")
    assert out.paraphrases == [] and out.hyde == "h"


async def test_failure_falls_back_silently():
    out = await Rewriter(FakeLLM(raise_exc=True)).rewrite("q", "both")
    assert out.fell_back is True and out.paraphrases == [] and out.hyde is None
