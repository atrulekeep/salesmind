"use client";

import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { ChatSettings, RetrievalMode, RewriteMode } from "@/lib/types";

const MODE_LABEL: Record<RetrievalMode, string> = {
  hybrid: "混合（BM25 + 向量）",
  vector: "纯向量",
};

const REWRITE_LABEL: Record<RewriteMode, string> = {
  none: "不改写",
  multi: "多查询",
  hyde: "HyDE",
  both: "多查询 + HyDE",
};

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <Label className="text-xs font-normal text-muted-foreground">
        {label}
      </Label>
      {children}
    </div>
  );
}

export function ChatSettingsPanel({
  settings,
  onChange,
  disabled,
}: {
  settings: ChatSettings;
  onChange: (next: ChatSettings) => void;
  disabled?: boolean;
}) {
  return (
    <div className="grid grid-cols-1 gap-2.5 rounded-lg border bg-card p-3 sm:grid-cols-2">
      <Field label="检索模式">
        <Select
          value={settings.mode}
          onValueChange={(v) =>
            v && onChange({ ...settings, mode: v as RetrievalMode })
          }
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(Object.keys(MODE_LABEL) as RetrievalMode[]).map((m) => (
              <SelectItem key={m} value={m}>
                {MODE_LABEL[m]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <Field label="Top K">
        <Select
          value={String(settings.topk)}
          onValueChange={(v) =>
            v != null && onChange({ ...settings, topk: Number(v) })
          }
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {[1, 2, 3, 4, 5, 6, 8, 10, 15, 20].map((k) => (
              <SelectItem key={k} value={String(k)}>
                {k}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <Field label="Rerank 重排">
        <Switch
          checked={settings.use_rerank}
          onCheckedChange={(v) => onChange({ ...settings, use_rerank: v })}
          disabled={disabled}
        />
      </Field>

      <Field label="查询改写">
        <Select
          value={settings.rewrite_mode}
          onValueChange={(v) =>
            v && onChange({ ...settings, rewrite_mode: v as RewriteMode })
          }
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(Object.keys(REWRITE_LABEL) as RewriteMode[]).map((r) => (
              <SelectItem key={r} value={r}>
                {REWRITE_LABEL[r]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
    </div>
  );
}
