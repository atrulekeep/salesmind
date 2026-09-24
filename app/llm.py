"""Ollama chat 客户端：JSON 模式与流式事件；OpenAI 兼容通道后续在此抽象。"""
import json
import re

import httpx


class LLMError(RuntimeError):
    pass


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _parse_json_content(content: str) -> dict:
    text = content.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model did not return valid JSON: {text[:120]}") from e
    if not isinstance(data, dict):
        raise LLMError("json model output must be an object")
    return data


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0,
                 client: httpx.AsyncClient | None = None,
                 think: bool | None = None):
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model
        self._timeout = timeout
        self._client = client
        # None=跟随模型默认；True/False=显式控制 qwen3 系 thinking
        self._think = think

    def _own_client(self) -> httpx.AsyncClient:
        # trust_env=False：Ollama 永远是本机地址，不能走系统/环境代理
        # （后台任务宿主注入的 HTTP(S)_PROXY 会把 127.0.0.1 请求转发到代理导致 ReadTimeout）
        if self._client:
            return self._client
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            trust_env=False,
        )

    async def chat_json(self, system: str, user: str, temperature: float = 0.1) -> dict:
        client = self._own_client()
        try:
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            }
            if self._think is not None:
                payload["think"] = self._think
            resp = await client.post(self._url, json=payload)
            resp.raise_for_status()
            return _parse_json_content(resp.json()["message"]["content"])
        except httpx.HTTPError as e:
            raise LLMError(f"chat_json failed: {e}") from e
        finally:
            if self._client is None:
                await client.aclose()

    async def stream(self, messages: list[dict], temperature: float = 0.3):
        client = self._own_client()
        try:
            payload = {
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            }
            if self._think is not None:
                payload["think"] = self._think
            req = client.build_request("POST", self._url, json=payload)
            resp = await client.send(req, stream=True)
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                evt = json.loads(line)
                if evt.get("done"):
                    yield {
                        "type": "done",
                        "usage": {
                            "prompt_tokens": evt.get("prompt_eval_count", 0),
                            "completion_tokens": evt.get("eval_count", 0),
                        },
                    }
                    break
                text = (evt.get("message") or {}).get("content", "")
                if text:
                    yield {"type": "delta", "text": text}
        finally:
            if self._client is None:
                await client.aclose()
