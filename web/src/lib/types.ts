/** 后端 SSE / API 契约类型（对应 app/api/routes.py、documents.py）。 */

export type RetrievalMode = "vector" | "hybrid";
export type RewriteMode = "none" | "multi" | "hyde" | "both";

export interface ChatSettings {
  mode: RetrievalMode;
  use_rerank: boolean;
  rewrite_mode: RewriteMode;
  topk: number;
  acl: string;
}

export const DEFAULT_SETTINGS: ChatSettings = {
  mode: "hybrid",
  use_rerank: true,
  rewrite_mode: "none",
  topk: 5,
  acl: "public",
};

/** retrieved 事件 items[] 元素；[n] 角标编号 n 对应 items[n-1]。 */
export interface RetrievedItem {
  chunk_id: string;
  title: string;
  source: string;
  section_path: string | null;
  valid_from: string | null;
  doc_version: string | null;
  chunk_type: string | null;
  content: string;
  final_rank: number | null;
  bm25_rank: number | null;
  vector_rank: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface ChatTrace {
  timings_ms: Record<string, number>;
  fused: boolean;
  rerank_available: boolean;
  rewrite_fell_back: boolean;
}

/** retrieving：检索中；generating：出字中；retrying：引用自纠，已清空重收；done/error：定格。 */
export type TurnPhase =
  | "retrieving"
  | "generating"
  | "retrying"
  | "done"
  | "error";

export interface ChatTurn {
  id: string;
  question: string;
  settings: ChatSettings;
  phase: TurnPhase;
  retryDetail: string | null;
  answer: string;
  items: RetrievedItem[];
  rerankAvailable: boolean;
  rewriteFellBack: boolean;
  citations: string[];
  citationOk: boolean | null;
  refused: boolean;
  usage: ChatUsage | null;
  latencyMs: number | null;
  trace: ChatTrace | null;
  error: string | null;
}

/** GET /api/documents 列表项与索引报告（对应 app/api/documents.py）。 */
export type DocStatus = "synced" | "stale" | "pending" | "missing";

export interface DocumentInfo {
  doc_id: string;
  title: string;
  file: string;
  md5: string;
  version: string | null;
  valid_from: string | null;
  valid_until: string | null;
  chunk_count: number;
  indexed_at: string | null;
  status: DocStatus;
}

export interface DocumentsList {
  items: DocumentInfo[];
  total_chunks: number;
}

export interface IndexReport {
  added: number;
  rebuilt: number;
  skipped: number;
  removed: number;
  total_chunks: number;
}

export function newTurn(question: string, settings: ChatSettings): ChatTurn {
  return {
    id:
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : String(Date.now()),
    question,
    settings,
    phase: "retrieving",
    retryDetail: null,
    answer: "",
    items: [],
    rerankAvailable: false,
    rewriteFellBack: false,
    citations: [],
    citationOk: null,
    refused: false,
    usage: null,
    latencyMs: null,
    trace: null,
    error: null,
  };
}
