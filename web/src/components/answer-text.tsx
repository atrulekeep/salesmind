"use client";

import { Fragment, type ReactNode } from "react";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { RetrievedItem } from "@/lib/types";

const CITATION_RE = /\[(\d+)\]/g;

function MetaLine({ label, value }: { label: string; value: ReactNode }) {
  if (value == null || value === "") return null;
  return (
    <div className="flex gap-1.5 text-xs">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words">{value}</span>
    </div>
  );
}

function SourceCard({ n, item }: { n: number; item: RetrievedItem }) {
  return (
    <div className="w-80">
      <PopoverTitle className="text-sm leading-snug">{item.title}</PopoverTitle>
      <PopoverDescription className="mt-0.5 text-xs">
        来源 [{n}]
        {item.section_path ? ` · ${item.section_path}` : ""}
      </PopoverDescription>
      <Separator className="my-2" />
      <div className="flex flex-col gap-1">
        <MetaLine label="文件" value={item.source} />
        <MetaLine label="版本" value={item.doc_version} />
        <MetaLine label="生效日期" value={item.valid_from} />
        <MetaLine label="类型" value={item.chunk_type} />
        <MetaLine label="chunk_id" value={item.chunk_id} />
      </div>
      <Separator className="my-2" />
      <ScrollArea className="max-h-60 pr-3">
        <p className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
          {item.content}
        </p>
      </ScrollArea>
    </div>
  );
}

/**
 * 渲染答案文本，把 [n] 切分成可点击角标；编号 n 对应 items[n-1]。
 * 越界编号（理论上不应出现）原样显示为文本。
 */
export function AnswerText({
  text,
  items,
}: {
  text: string;
  items: RetrievedItem[];
}) {
  const parts: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  CITATION_RE.lastIndex = 0;
  let key = 0;
  while ((m = CITATION_RE.exec(text)) !== null) {
    if (m.index > last) {
      parts.push(text.slice(last, m.index));
    }
    const n = Number(m[1]);
    const item = items[n - 1];
    if (item) {
      parts.push(
        <Popover key={`cite-${key++}`}>
          <PopoverTrigger
            render={
              <button
                type="button"
                className="mx-0.5 inline-flex h-[1.15rem] min-w-[1.15rem] items-center justify-center rounded border bg-muted px-1 align-baseline text-[0.7rem] font-medium text-primary transition-colors hover:bg-accent"
                aria-label={`查看来源 ${n}`}
              />
            }
          >
            {n}
          </PopoverTrigger>
          <PopoverContent align="start" sideOffset={6}>
            <SourceCard n={n} item={item} />
          </PopoverContent>
        </Popover>,
      );
    } else {
      parts.push(<Fragment key={`cite-x-${key++}`}>{m[0]}</Fragment>);
    }
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));

  return (
    <div className="whitespace-pre-wrap break-words text-sm leading-7">
      {parts}
    </div>
  );
}
