from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    content: str
    raw_content: str
    chunk_index: int
    chunk_type: str  # markdown | table | recursive
    title: str
    section_path: str
    source: str
    doc_version: str
    valid_from: str | None
    valid_until: str | None
    acl: str
    content_hash: str


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_rank: int
