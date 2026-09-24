"use client";

import { useCallback, useRef, useState } from "react";
import type { ChatSettings, ChatTurn } from "@/lib/types";
import { newTurn } from "@/lib/types";

interface SSEFrame {
  event: string;
  // 后端各事件 data 结构不同，在消费处按 event 收窄
  data: Record<string, unknown>;
}

/** 按 SSE 帧边界（空行）切分；返回完整帧与剩余半包。 */
export function parseSSE(buffer: string): {
  frames: SSEFrame[];
  rest: string;
} {
  const normalized = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  const frames: SSEFrame[] = [];
  for (const part of parts) {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of part.split("\n")) {
      if (line.startsWith(":")) continue; // 注释/心跳
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length === 0) continue;
    try {
      frames.push({ event, data: JSON.parse(dataLines.join("\n")) });
    } catch {
      // 半包/非法 JSON 丢弃，等下一 chunk 拼包（正常不会发生）
    }
  }
  return { frames, rest };
}

function asString(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

/**
 * 消费 POST SSE：EventSource 不支持 POST，故 fetch + ReadableStream 手动解析。
 * 状态机：status(retrieving) → retrieved → answer_delta*
 *   → 可选 status(citation_retry)（清空 answer 重收）→ done。
 */
export function useChatStream() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patchTurn = useCallback(
    (id: string, p: Partial<ChatTurn> | ((t: ChatTurn) => ChatTurn)) => {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id ? (typeof p === "function" ? p(t) : { ...t, ...p }) : t,
        ),
      );
    },
    [],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const ask = useCallback(
    async (rawQuestion: string, settings: ChatSettings) => {
      const question = rawQuestion.trim();
      if (!question) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const turn = newTurn(question, settings);
      setTurns((prev) => [...prev, turn]);
      setStreaming(true);

      const fail = (message: string) =>
        patchTurn(turn.id, (t) =>
          t.phase === "done"
            ? t
            : { ...t, phase: "error", error: message },
        );

      let resp: Response;
      try {
        resp = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question, ...settings }),
          signal: controller.signal,
        });
      } catch (err) {
        if (controller.signal.aborted) {
          fail("已停止");
        } else {
          fail(err instanceof Error ? err.message : "网络错误");
        }
        if (abortRef.current === controller) {
          abortRef.current = null;
          setStreaming(false);
        }
        return;
      }

      if (!resp.ok || !resp.body) {
        let message = `HTTP ${resp.status}`;
        try {
          const j = (await resp.json()) as { detail?: unknown };
          if (typeof j.detail === "string") message = j.detail;
          else if (j.detail != null) message = JSON.stringify(j.detail);
        } catch {
          // 非 JSON 错误体，保留状态码
        }
        fail(message);
        if (abortRef.current === controller) {
          abortRef.current = null;
          setStreaming(false);
        }
        return;
      }

      try {
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        const dispatch = (frame: SSEFrame) => {
          const d = frame.data;
          switch (frame.event) {
            case "status": {
              const stage = asString(d.stage);
              if (stage === "retrieving") {
                patchTurn(turn.id, { phase: "retrieving" });
              } else if (stage === "citation_retry") {
                // 引用自纠：清空已渲染文本，重新接收 answer_delta
                patchTurn(turn.id, {
                  phase: "retrying",
                  retryDetail: asString(d.detail),
                  answer: "",
                });
              }
              break;
            }
            case "retrieved":
              patchTurn(turn.id, {
                items: (d.items as ChatTurn["items"]) ?? [],
                rerankAvailable: d.rerank_available === true,
                rewriteFellBack: d.rewrite_fell_back === true,
                phase: "generating",
              });
              break;
            case "answer_delta": {
              const text = asString(d.text) ?? "";
              patchTurn(turn.id, (t) => ({
                ...t,
                phase: "generating",
                answer: t.answer + text,
              }));
              break;
            }
            case "done":
              patchTurn(turn.id, {
                phase: "done",
                answer: asString(d.answer) ?? "",
                citations: Array.isArray(d.citations)
                  ? (d.citations as unknown[]).filter(
                      (n): n is string => typeof n === "string",
                    )
                  : [],
                citationOk:
                  typeof d.citation_ok === "boolean"
                    ? d.citation_ok
                    : null,
                refused: d.refused === true,
                usage: (d.usage as ChatTurn["usage"]) ?? null,
                latencyMs:
                  typeof d.latency_ms === "number" ? d.latency_ms : null,
                trace: (d.trace as ChatTurn["trace"]) ?? null,
              });
              break;
            default:
              break;
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const parsed = parseSSE(buf);
          buf = parsed.rest;
          for (const frame of parsed.frames) dispatch(frame);
        }
        buf += decoder.decode();
        const tail = parseSSE(buf);
        for (const frame of tail.frames) dispatch(frame);

        // 未收到 done 流就结束：连接异常中断
        patchTurn(turn.id, (t) =>
          t.phase === "done" || controller.signal.aborted
            ? t
            : { ...t, phase: "error", error: t.error ?? "连接中断，未收到完成事件" },
        );
        if (controller.signal.aborted) fail("已停止");
      } catch (err) {
        if (controller.signal.aborted) {
          fail("已停止");
        } else {
          fail(err instanceof Error ? err.message : "读取流失败");
        }
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
          setStreaming(false);
        }
      }
    },
    [patchTurn],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setTurns([]);
    setStreaming(false);
  }, []);

  return { turns, streaming, ask, stop, reset };
}
