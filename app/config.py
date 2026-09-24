from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SALESMIND_", extra="ignore")

    provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:9b"
    # qwen3 系 thinking 默认关闭（开启时单题延迟 1.4s→57s 级）；SALESMIND_LLM_THINK=true 可开
    llm_think: bool = False
    embedding_model: str = "bge-large:latest"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    hf_endpoint: str = "https://hf-mirror.com"

    corpus_dir: str = "data/corpus"
    index_dir: str = "data/index"
    chunk_size: int = 512
    chunk_overlap: int = 64
    rrf_k: int = 60
    topn_recall: int = 20
    topk_final: int = 5
    embed_batch: int = 16
    request_timeout: float = 180.0
    default_acl: str = "public"

    @property
    def chroma_path(self) -> str:
        return f"{self.index_dir}/chroma"

    @property
    def chunks_path(self) -> str:
        return f"{self.index_dir}/chunks.jsonl"

    @property
    def manifest_path(self) -> str:
        return f"{self.index_dir}/manifest.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
