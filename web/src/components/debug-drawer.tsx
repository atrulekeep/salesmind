"use client";

import { AlertTriangle, Layers } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "cn";
import type { ChatTurn, RetrievedItem } from "@/lib/types";

const TIMING_LABEL: Record<string, string> = {
  rewrite: "查询改写",
  vector_recall: "向量召回",
  bm25_recall: "BM25 召回",
  rerank: "rerank 重排",
  total: "总计",
};

function TimingBars({ timings }: { timings: Record<string, number> }) {
  const entries = Object.entries(timings);
  if (entries.length === 0) {
    return <p className="text-xs text-muted-foreground">无耗时数据</p>;
  }
  const max = Math.max(...entries.map(([, v]) => v), 1);
  return (
    <div className="flex flex-col gap-1.5">
      {entries.map(([key, ms]) => (
        <div key={key} className="flex items-center gap-2 text-xs">
          <span className="w-20 shrink-0 text-right text-muted-foreground">
            {TIMING_LABEL[key] ?? key}
          </span>
          <div className="h-3.5 flex-1 overflow-hidden rounded bg-muted">
            <div
              className={cn(
                "h-full rounded",
                key === "total" ? "bg-primary/70" : "bg-primary/40",
              )}
              style={{ width: `${Math.max((ms / max) * 100, 2)}%` }}
            />
          </div>
          <span className="w-16 shrink-0 tabular-nums">{ms.toFixed(0)} ms</span>
        </div>
      ))}
    </div>
  );
}

function Stat({
  label,
  children,
  dim,
}: {
  label: string;
  children: React.ReactNode;
  dim?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center gap-0.5 rounded border px-1 py-1.5",
        dim ? "border-dashed text-muted-foreground/60" : "bg-muted/50",
      )}
    >
      <span className="text-[10px] text-muted-foreground">{label}</span>
      <span className="text-xs font-medium tabular-nums">{children}</span>
    </div>
  );
}

/** 相对最终名次的变化：正=上升，负=下降。 */
function RankDelta({ baseline, finalRank }: { baseline: number | null; finalRank: number | null }) {
  if (baseline == null || finalRank == null || baseline === finalRank) {
    return null;
  }
  const up = baseline > finalRank;
  return (
    <span
      className={cn(
        "ml-1 text-[10px]",
        up ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400",
      )}
    >
      {up ? "↑" : "↓"}
      {Math.abs(baseline - finalRank)}
    </span>
  );
}

function ItemCard({ item, rerankUsed }: { item: RetrievedItem; rerankUsed: boolean }) {
  return (
    <div className="rounded-lg border p-2.5">
      <div className="flex items-start gap-2">
        <Badge variant="secondary" className="mt-0.5 shrink-0">
          #{item.final_rank ?? "?"}
        </Badge>
        <div className="min-w-0">
          <p className="truncate text-xs font-medium" title={item.title}>
            {item.title}
          </p>
          <p className="truncate text-[10px] text-muted-foreground">
            {item.section_path ? `${item.section_path} · ` : ""}
            {item.source}
          </p>
        </div>
      </div>
      <div className="mt-2 grid grid-cols-4 gap-1.5">
        <Stat label="BM25 名次" dim={item.bm25_rank == null}>
          {item.bm25_rank != null ? (
            <>
              #{item.bm25_rank}
              <RankDelta baseline={item.bm25_rank} finalRank={item.final_rank} />
            </>
          ) : (
            "—"
          )}
        </Stat>
        <Stat label="向量名次" dim={item.vector_rank == null}>
          {item.vector_rank != null ? (
            <>
              #{item.vector_rank}
              <RankDelta baseline={item.vector_rank} finalRank={item.final_rank} />
            </>
          ) : (
            "—"
          )}
        </Stat>
        <Stat label="RRF 分" dim={item.rrf_score == null}>
          {item.rrf_score != null ? item.rrf_score.toFixed(4) : "—"}
        </Stat>
        <Stat label="rerank 分" dim={!rerankUsed || item.rerank_score == null}>
          {rerankUsed && item.rerank_score != null
            ? item.rerank_score.toFixed(4)
            : "—"}
        </Stat>
      </div>
      <p className="mt-1.5 truncate font-mono text-[10px] text-muted-foreground/70">
        {item.chunk_id}
      </p>
    </div>
  );
}

export function DebugDrawer({
  turns,
  turnId,
  open,
  onOpenChange,
  onSelectTurn,
}: {
  turns: ChatTurn[];
  turnId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectTurn: (id: string | null) => void;
}) {
  const turn = turns.find((t) => t.id === turnId) ?? null;
  const items = [...(turn?.items ?? [])].sort(
    (a, b) => (a.final_rank ?? 999) - (b.final_rank ?? 999),
  );
  const rerankUsed =
    turn != null &&
    turn.settings.use_rerank &&
    turn.rerankAvailable &&
    items.some((it) => it.rerank_score != null);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full gap-0 p-0 data-[side=right]:sm:max-w-xl"
      >
        <SheetHeader className="border-b">
          <SheetTitle className="flex items-center gap-1.5">
            <Layers className="size-4" />
            检索调试
          </SheetTitle>
          <SheetDescription className="truncate">
            {turn ? turn.question : "暂无数据"}
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {!turn ? (
            <p className="text-sm text-muted-foreground">暂无检索数据</p>
          ) : (
            <>
              {turns.length > 1 && (
                <Select
                  value={turn.id}
                  onValueChange={(v) => onSelectTurn(v)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {turns.map((t, i) => (
                      <SelectItem key={t.id} value={t.id}>
                        第 {i + 1} 轮：{t.question.slice(0, 20)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}

              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                <Badge variant="outline">
                  {turn.settings.mode === "hybrid" ? "混合检索" : "纯向量"} · top
                  {turn.settings.topk}
                </Badge>
                <Badge variant="outline">
                  改写：{turn.settings.rewrite_mode}
                </Badge>
                {turn.trace?.fused && <Badge variant="secondary">RRF 融合</Badge>}
                {rerankUsed ? (
                  <Badge variant="secondary">rerank 已重排</Badge>
                ) : (
                  <Badge variant="outline" className="text-muted-foreground">
                    rerank 未生效
                  </Badge>
                )}
              </div>
              {turn.rewriteFellBack && (
                <div className="flex items-start gap-1.5 rounded-md bg-amber-50 px-2.5 py-1.5 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
                  <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                  查询改写失败，已回退为原问题检索
                </div>
              )}

              <section className="space-y-2">
                <h3 className="text-xs font-medium text-muted-foreground">
                  阶段耗时
                </h3>
                {turn.trace ? (
                  <TimingBars timings={turn.trace.timings_ms} />
                ) : (
                  <p className="text-xs text-muted-foreground">
                    回答完成后展示
                  </p>
                )}
              </section>

              <section className="space-y-2">
                <h3 className="text-xs font-medium text-muted-foreground">
                  命中块（{items.length}）
                </h3>
                {items.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    {turn.phase === "retrieving" ? "检索中…" : "未召回任何块"}
                  </p>
                ) : (
                  <div className="flex flex-col gap-2">
                    {items.map((it) => (
                      <ItemCard
                        key={it.chunk_id}
                        item={it}
                        rerankUsed={rerankUsed}
                      />
                    ))}
                  </div>
                )}
              </section>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
