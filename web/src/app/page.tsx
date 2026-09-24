"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  FileText,
  SendHorizonal,
  SlidersHorizontal,
  Square,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatSettingsPanel } from "@/components/chat-settings";
import { ChatTurnView } from "@/components/chat-turn";
import { DebugDrawer } from "@/components/debug-drawer";
import { useChatStream } from "@/hooks/use-chat-stream";
import { DEFAULT_SETTINGS, type ChatSettings } from "@/lib/types";

export default function Home() {
  const { turns, streaming, ask, stop, reset } = useChatStream();
  const [input, setInput] = useState("");
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [debugTurnId, setDebugTurnId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || streaming) return;
    const q = input;
    setInput("");
    void ask(q, settings);
  };

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col px-4">
      <header className="flex items-center justify-between border-b py-3">
        <div>
          <h1 className="text-base font-semibold">SalesMind 知识问答</h1>
          <p className="text-xs text-muted-foreground">
            基于内部资料的 RAG 问答，回答附可核验的来源引用
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          render={<Link href="/documents" />}
        >
          <FileText className="size-4" />
          文档管理
        </Button>
      </header>

      <main className="flex-1 space-y-5 overflow-y-auto py-5">
        {turns.length === 0 ? (
          <div className="mt-24 flex flex-col items-center gap-2 text-center text-sm text-muted-foreground">
            <p className="text-foreground text-base font-medium">
              问一个资料库里有的问题
              <br />
              例如：产品支持哪些部署方式？
            </p>
            <p className="text-xs">回答中的 [n] 角标可点击查看原文出处</p>
          </div>
        ) : (
          turns.map((turn) => (
            <ChatTurnView
              key={turn.id}
              turn={turn}
              onDebug={() => {
                setDebugTurnId(turn.id);
                setDebugOpen(true);
              }}
            />
          ))
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="border-t py-3">
        {showSettings && (
          <div className="mb-2.5">
            <ChatSettingsPanel
              settings={settings}
              onChange={setSettings}
              disabled={streaming}
            />
          </div>
        )}
        <form onSubmit={submit} className="flex items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="icon-lg"
            className="shrink-0"
            onClick={() => setShowSettings((v) => !v)}
            aria-label="检索设置"
            title="检索设置"
          >
            <SlidersHorizontal className="size-4" />
          </Button>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={streaming ? "回答生成中…" : "输入问题，Enter 发送"}
            disabled={streaming}
            className="h-9"
          />
          {streaming ? (
            <Button
              type="button"
              size="icon-lg"
              variant="destructive"
              className="shrink-0"
              onClick={stop}
              aria-label="停止"
              title="停止生成"
            >
              <Square className="size-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon-lg"
              className="shrink-0"
              disabled={!input.trim()}
              aria-label="发送"
            >
              <SendHorizonal className="size-4" />
            </Button>
          )}
          {turns.length > 0 && !streaming && (
            <Button
              type="button"
              variant="ghost"
              size="icon-lg"
              className="shrink-0"
              onClick={reset}
              aria-label="清空对话"
              title="清空对话"
            >
              <Trash2 className="size-4" />
            </Button>
          )}
        </form>
      </footer>

      <DebugDrawer
        turns={turns}
        turnId={debugTurnId}
        open={debugOpen}
        onOpenChange={setDebugOpen}
        onSelectTurn={setDebugTurnId}
      />
    </div>
  );
}
