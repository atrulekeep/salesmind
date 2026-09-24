"""Ollama 嵌入客户端：/api/embed 批量调用，重试与维度断言。"""
import asyncio
import json

import httpx


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int = 1024,
        batch: int = 16,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model
        self._dim = dimension
        self._batch = batch
        self._timeout = timeout
        self._client = client

    def _own_client(self) -> httpx.AsyncClient:
        # trust_env=False：本机 Ollama 不走环境代理，避免代理转发导致超时
        if self._client:
            return self._client
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            trust_env=False,
        )

    async def _post(self, client: httpx.AsyncClient, inputs: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = await client.post(
                    self._url, json={"model": self._model, "input": inputs}
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    raise EmbeddingError(f"embed service error: HTTP {resp.status_code}")
                resp.raise_for_status()  # 4xx 参数错误：立即终止，不重试
                vectors = json.loads(resp.content)["embeddings"]
                if any(len(v) != self._dim for v in vectors):
                    raise EmbeddingError(
                        f"embedding dim mismatch: expect {self._dim}, "
                        f"got {[len(v) for v in vectors][:3]}"
                    )
                return vectors
            except httpx.TransportError as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
        raise EmbeddingError(f"embed request failed after retries: {last_err}")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._own_client()
        try:
            out: list[list[float]] = []
            for i in range(0, len(texts), self._batch):
                out.extend(await self._post(client, texts[i : i + self._batch]))
            return out
        finally:
            if self._client is None:
                await client.aclose()

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
