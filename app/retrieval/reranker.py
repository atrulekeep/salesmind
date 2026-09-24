"""bge-reranker-v2-m3 本地交叉编码器。transformers/torch 为可选依赖，缺失即降级。"""
import os

from app.schemas import Chunk


class BGEReranker:
    def __init__(self, model_name: str, hf_endpoint: str | None = None):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._tried = False
        self._available = False
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    def available(self) -> bool:
        if not self._tried:
            self._try_load()
        return self._available

    @property
    def status(self) -> str:
        """health 探针用：不触发加载。lazy=未尝试；available/unavailable=已尝试。"""
        if not self._tried:
            return "lazy"
        return "available" if self._available else "unavailable"

    def _try_load(self) -> None:
        self._tried = True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            ).to(self._device).eval()
            self._available = True
        except Exception:  # noqa: BLE001 - 模型未下载/依赖缺失/设备不支持统一降级
            self._available = False

    def rerank(
        self, query: str, chunks: list[Chunk], topk: int
    ) -> list[tuple[str, float]] | None:
        if not chunks:
            return []
        if not self.available():
            return None
        import torch

        pairs = [[query, c.content] for c in chunks]
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self._device)
            logits = self._model(**inputs).logits.squeeze(-1).float().cpu().tolist()
        if isinstance(logits, float):
            logits = [logits]
        ordered = sorted(
            ((chunks[i].chunk_id, float(s)) for i, s in enumerate(logits)),
            key=lambda x: x[1], reverse=True,
        )
        return ordered[:topk]
