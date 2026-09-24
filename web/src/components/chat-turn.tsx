"use client";

import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  Loader2,
  ListOrdered,
  RefreshCw,
  XCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AnswerText } from "@/components/answer-text";
import { cn } from "cn";
import type { ChatTurn } from "@/lib/types";

const REFUSAL_TEXT = "根据现有资料未找到相关信息。";

function PhaseHint({ turn }: { turn: ChatTurn }) {
  if (turn.phase === "retrieving") {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        正在检索资料…
      </div>
    );
  }
  if (turn.phase === "generating" && turn.answer === "") {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        正在生成回答…
      </div>
    );
  }
  if (turn.phase === "retrying") {
    return (
      <div className="flex items-start gap-2 rounded-md bg-amber-50 px-2.5 py-2 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
        <RefreshCw className="mt-0.5 size-3.5 shrink-0 animate-spin" />
        <div>
          <p className="font-medium">引用校验未通过，正在重新检索并生成…</p>
          {turn.retryDetail && (
            <p className="mt-0.5 break-words text-amber-700/80 dark:text-amber-400/80">
              {turn.retryDetail}
            </p>
          )}
        </div>
      </div>
    );
  }
  return null;
}

function MetaBar({ turn }: { turn: ChatTurn }) {
  if (turn.phase !== "done") return null;
  const items: React.ReactNode[] = [];
  if (turn.latencyMs != null) {
    items.push(
      <span key="latency" className="text-muted-foreground">
        耗时 {(turn.latencyMs / 1000).toFixed(1)}s
      </span>,
    );
  }
  if (turn.usage) {
    items.push(
      <span key="tokens" className="text-muted-foreground">
        tokens {turn.usage.prompt_tokens}/{turn.usage.completion_tokens}
      </span>,
    );
  }
  if (turn.refused) {
    items.push(
      <Badge key="refused" variant="secondary" className="gap-1">
        <Ban className="size-3" />
        已拒答
      </Badge>,
    );
  } else if (turn.citationOk === true) {
    items.push(
      <Badge key="cite-ok" variant="secondary" className="gap-1">
        <CheckCircle2 className="size-3 text-emerald-600" />
        引用校验通过
      </Badge>,
    );
  } else if (turn.citationOk === false) {
    items.push(
      <Badge key="cite-bad" variant="destructive" className="gap-1">
        <AlertTriangle className="size-3" />
        引用校验未通过
      </Badge>,
    );
  }
  if (turn.settings.use_rerank && !turn.rerankAvailable) {
    items.push(
      <span key="norerank" className="text-xs text-amber-700 dark:text-amber-400">
        rerank 不可用，已跳过重排
      </span>,
    );
  }
  if (turn.rewriteFellBack) {
    items.push(
      <span key="rewritefb" className="text-xs text-amber-700 dark:text-amber-400">
        查询改写失败，已回退原问题
      </span>,
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      {items}
    </div>
  );
}

export function ChatTurnView({
  turn,
  onDebug,
}: {
  turn: ChatTurn;
  onDebug?: () => void;
}) {
  const streaming = turn.phase === "generating";
  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-3.5 py-2 text-sm text-primary-foreground">
          <p className="whitespace-pre-wrap break-words">{turn.question}</p>
        </div>
      </div>

      <div className="flex justify-start">
        <div
          className={cn(
            "flex max-w-full flex-1 flex-col gap-2 rounded-2xl rounded-bl-sm border px-3.5 py-3",
            turn.refused && turn.phase === "done"
              ? "border-border bg-muted/60 text-muted-foreground"
              : "bg-card",
            turn.phase === "error" && "border-destructive/40",
          )}
        >
          <PhaseHint turn={turn} />

          {turn.answer !== "" &&
            (turn.refused && turn.phase === "done" ? (
              <p className="text-sm italic text-muted-foreground">
                {REFUSAL_TEXT}
              </p>
            ) : (
              <div>
                <AnswerText text={turn.answer} items={turn.items} />
                {streaming && (
                  <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-foreground align-middle" />
                )}
              </div>
            ))}

          {turn.phase === "error" && (
            <div className="flex items-start gap-2 text-xs text-destructive">
              <XCircle className="mt-0.5 size-3.5 shrink-0" />
              <span className="break-words">{turn.error ?? "请求失败"}</span>
            </div>
          )}

          <MetaBar turn={turn} />

          {onDebug && turn.items.length > 0 && (
            <div>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                className="text-muted-foreground"
                onClick={onDebug}
              >
                <ListOrdered className="size-3" />
                检索调试（{turn.items.length} 块）
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
