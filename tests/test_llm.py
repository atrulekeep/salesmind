import json

import httpx
import pytest

from app.llm import LLMError, OllamaLLM


def client_with(lines=None, obj=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if obj is not None:
            return httpx.Response(200, json=obj)
        body = ("\n".join(json.dumps(l) for l in lines)).encode()
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")


async def test_chat_json_parses_fenced_content():
    obj = {"message": {"content": '```json\n{"paraphrases": ["x"]}\n```'}}
    llm = OllamaLLM("http://ollama", "q", client=client_with(obj=obj))
    out = await llm.chat_json("sys", "user")
    assert out == {"paraphrases": ["x"]}


async def test_chat_json_invalid_raises():
    obj = {"message": {"content": "not json at all"}}
    llm = OllamaLLM("http://ollama", "q", client=client_with(obj=obj))
    with pytest.raises(LLMError):
        await llm.chat_json("sys", "user")


async def test_stream_emits_deltas_and_done_usage():
    lines = [
        {"message": {"role": "assistant", "content": "你"}, "done": False},
        {"message": {"role": "assistant", "content": "好"}, "done": False},
        {"message": {"content": ""}, "done": True,
         "prompt_eval_count": 100, "eval_count": 2},
    ]
    llm = OllamaLLM("http://ollama", "q", client=client_with(lines=lines))
    events = [
        ev
        async for ev in llm.stream([{"role": "user", "content": "hi"}])
    ]
    assert events[:2] == [
        {"type": "delta", "text": "你"},
        {"type": "delta", "text": "好"},
    ]
    assert events[2]["type"] == "done"
    assert events[2]["usage"] == {"prompt_tokens": 100, "completion_tokens": 2}
