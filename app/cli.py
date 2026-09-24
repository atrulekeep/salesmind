"""SalesMind 命令行：build-index（增量构建）与 ask（端到端问答）。"""
import argparse
import asyncio
import os

from app.config import get_settings
from app.generation.prompt import build_messages
from app.indexing.bm25_store import BM25Store
from app.indexing.embeddings import OllamaEmbedder
from app.indexing.manifest import Manifest
from app.indexing.pipeline import IndexPipeline
from app.indexing.vector_store import VectorStore
from app.llm import OllamaLLM
from app.retrieval.reranker import BGEReranker
from app.retrieval.retriever import Retriever
from app.retrieval.rewrite import Rewriter


def _build_components():
    s = get_settings()
    embedder = OllamaEmbedder(
        s.ollama_base_url, s.embedding_model, dimension=s.embedding_dim,
        batch=s.embed_batch, timeout=s.request_timeout,
    )
    vs = VectorStore(s.chroma_path)
    bm = BM25Store(s.chunks_path)
    bm.load()
    mf = Manifest(s.manifest_path)
    llm = OllamaLLM(s.ollama_base_url, s.llm_model, timeout=s.request_timeout,
                    think=s.llm_think)
    reranker = BGEReranker(s.reranker_model, s.hf_endpoint)
    return s, embedder, vs, bm, mf, llm, reranker


async def cmd_build(full: bool) -> None:
    s = get_settings()
    if full:
        import shutil

        # 只删索引产物，保留 data/index/.gitkeep
        for target in (s.chroma_path, s.chunks_path, s.manifest_path):
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.exists(target):
                os.remove(target)
    embedder = OllamaEmbedder(
        s.ollama_base_url, s.embedding_model, dimension=s.embedding_dim,
        batch=s.embed_batch, timeout=s.request_timeout,
    )
    vs = VectorStore(s.chroma_path)
    bm = BM25Store(s.chunks_path)
    mf = Manifest(s.manifest_path)
    report = await IndexPipeline(s, embedder, vs, bm, mf).run()
    print(report.model_dump())


async def cmd_ask(question: str, mode: str, use_rerank: bool, rewrite_mode: str) -> None:
    s, embedder, vs, bm, _, llm, reranker = _build_components()
    retriever = Retriever(
        embedder, vs, bm,
        reranker=reranker, rewriter=Rewriter(llm), settings=s,
    )
    result = await retriever.retrieve(
        question, mode=mode, use_rerank=use_rerank, rewrite_mode=rewrite_mode
    )
    print(f"\n检索：{len(result.items)} 块 | rerank={result.rerank_available} "
          f"| 耗时={result.trace.timings_ms['total']:.0f}ms")
    messages = build_messages(question, [it.chunk for it in result.items])
    print("回答：")
    usage = {}
    async for ev in llm.stream(messages):
        if ev["type"] == "delta":
            print(ev["text"], end="", flush=True)
        else:
            usage = ev["usage"]
    print("\n\n引用：")
    for it in result.items:
        rrf = f" rrf={it.rrf_score:.4f}" if it.rrf_score is not None else ""
        rk = f" rerank={it.rerank_score:.4f}" if it.rerank_score is not None else ""
        print(f"  [{it.final_rank}] {it.chunk.title} > {it.chunk.section_path} "
              f"({it.chunk.chunk_id}){rrf}{rk}")
    print(f"\ntoken：{usage}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="salesmind")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-index", help="构建/增量更新索引")
    p_build.add_argument("--full", action="store_true", help="清空后全量重建")

    p_ask = sub.add_parser("ask", help="端到端问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    p_ask.add_argument("--rerank", action="store_true")
    p_ask.add_argument("--rewrite", choices=["none", "multi", "hyde", "both"], default="none")

    args = parser.parse_args()
    if args.cmd == "build-index":
        asyncio.run(cmd_build(args.full))
    else:
        asyncio.run(cmd_ask(args.question, args.mode, args.rerank, args.rewrite))


if __name__ == "__main__":
    main()
