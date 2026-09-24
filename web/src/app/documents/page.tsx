"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import {
  ArrowLeft,
  FileUp,
  Loader2,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "cn";
import type {
  DocStatus,
  DocumentInfo,
  DocumentsList,
  IndexReport,
} from "@/lib/types";

const STATUS_UI: Record<
  DocStatus,
  { label: string; className: string }
> = {
  synced: {
    label: "已同步",
    className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300",
  },
  stale: {
    label: "文件变更，待重建",
    className: "bg-amber-100 text-amber-800 dark:bg-amber-950/50 dark:text-amber-300",
  },
  pending: {
    label: "新文件，未索引",
    className: "bg-sky-100 text-sky-800 dark:bg-sky-950/50 dark:text-sky-300",
  },
  missing: {
    label: "文件缺失",
    className: "bg-rose-100 text-rose-800 dark:bg-rose-950/50 dark:text-rose-300",
  },
};

interface ConflictInfo {
  conflict: "exists";
  doc_id: string;
  title: string;
  version: string | null;
  chunk_count: number;
}

function reportText(r: IndexReport): string {
  return `新增 ${r.added} · 重建 ${r.rebuilt} · 跳过 ${r.skipped} · 移除 ${r.removed}（共 ${r.total_chunks} 块）`;
}

async function errorDetail(resp: Response): Promise<string> {
  try {
    const j = (await resp.json()) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (j.detail != null) return JSON.stringify(j.detail);
  } catch {
    // 非 JSON
  }
  return `HTTP ${resp.status}`;
}

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  return iso.slice(0, 16).replace("T", " ");
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<DocumentInfo | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [conflict, setConflict] = useState<{
    file: File;
    info: ConflictInfo;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchList = useCallback(async () => {
    setLoading(true);
    setListError(null);
    try {
      const resp = await fetch("/api/documents", { cache: "no-store" });
      if (!resp.ok) throw new Error(await errorDetail(resp));
      const data = (await resp.json()) as DocumentsList;
      setDocs(data.items);
      setTotalChunks(data.total_chunks);
    } catch (err) {
      setListError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchList();
  }, [fetchList]);

  const upload = useCallback(
    async (file: File, overwrite: boolean) => {
      setUploading(true);
      const form = new FormData();
      form.append("file", file);
      try {
        const resp = await fetch(
          `/api/documents?overwrite=${overwrite ? "true" : "false"}`,
          { method: "POST", body: form },
        );
        if (resp.ok) {
          const j = (await resp.json()) as {
            doc_id: string;
            overwritten: boolean;
            report: IndexReport;
          };
          toast.success(
            `${j.overwritten ? "已覆盖" : "已上传"} ${j.doc_id}`,
            { description: reportText(j.report) },
          );
          setConflict(null);
          await fetchList();
          return;
        }
        if (resp.status === 409 && !overwrite) {
          const j = (await resp.json()) as { detail: ConflictInfo };
          setConflict({ file, info: j.detail });
          return;
        }
        toast.error("上传失败", { description: await errorDetail(resp) });
      } catch (err) {
        toast.error("上传失败", {
          description: err instanceof Error ? err.message : "网络错误",
        });
      } finally {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [fetchList],
  );

  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void upload(file, false);
  };

  const doDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const resp = await fetch(
        `/api/documents/${encodeURIComponent(deleteTarget.doc_id)}`,
        { method: "DELETE" },
      );
      if (!resp.ok) {
        toast.error("删除失败", { description: await errorDetail(resp) });
        return;
      }
      const j = (await resp.json()) as {
        removed: string;
        report: IndexReport;
      };
      toast.success(`已删除 ${j.removed}`, {
        description: reportText(j.report),
      });
      setDeleteTarget(null);
      await fetchList();
    } catch (err) {
      toast.error("删除失败", {
        description: err instanceof Error ? err.message : "网络错误",
      });
    } finally {
      setDeleting(false);
    }
  };

  const doReindex = async () => {
    setReindexing(true);
    try {
      const resp = await fetch("/api/documents/reindex", { method: "POST" });
      if (!resp.ok) {
        toast.error("重建失败", { description: await errorDetail(resp) });
        return;
      }
      const report = (await resp.json()) as IndexReport;
      toast.success("索引重建完成", { description: reportText(report) });
      await fetchList();
    } catch (err) {
      toast.error("重建失败", {
        description: err instanceof Error ? err.message : "网络错误",
      });
    } finally {
      setReindexing(false);
    }
  };

  return (
    <div className="mx-auto flex h-screen max-w-6xl flex-col px-4">
      <header className="flex items-center justify-between gap-3 border-b py-3">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon-sm"
            render={<Link href="/" />}
            aria-label="返回问答"
            title="返回问答"
          >
            <ArrowLeft className="size-4" />
          </Button>
          <div>
            <h1 className="text-base font-semibold">文档管理</h1>
            <p className="text-xs text-muted-foreground">
              {loading
                ? "加载中…"
                : `${docs.length} 篇文档 · 索引共 ${totalChunks} 块`}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={doReindex}
            disabled={reindexing || loading}
          >
            {reindexing ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
            重建索引
          </Button>
          <Button
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <FileUp className="size-4" />
            )}
            上传 .md
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".md,text/markdown"
            className="hidden"
            onChange={onPickFile}
          />
        </div>
      </header>

      <main className="flex-1 overflow-y-auto py-4">
        {listError ? (
          <div className="mt-24 flex flex-col items-center gap-3 text-sm">
            <p className="text-destructive">文档列表加载失败：{listError}</p>
            <Button variant="outline" size="sm" onClick={() => void fetchList()}>
              重试
            </Button>
          </div>
        ) : loading ? (
          <div className="mt-24 flex justify-center text-sm text-muted-foreground">
            <Loader2 className="mr-2 size-4 animate-spin" />
            加载中…
          </div>
        ) : docs.length === 0 ? (
          <div className="mt-24 text-center text-sm text-muted-foreground">
            语料库为空，点击右上角「上传 .md」添加第一篇文档
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>文档</TableHead>
                  <TableHead>版本</TableHead>
                  <TableHead>MD5</TableHead>
                  <TableHead className="text-right">块数</TableHead>
                  <TableHead>时效</TableHead>
                  <TableHead>索引时间</TableHead>
                  <TableHead>状态</TableHead>
                  <TableHead className="text-right">操作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {docs.map((d) => {
                  const status = STATUS_UI[d.status];
                  return (
                    <TableRow key={d.doc_id}>
                      <TableCell className="min-w-48">
                        <div className="font-medium">{d.title}</div>
                        <div className="font-mono text-[11px] text-muted-foreground">
                          {d.doc_id} · {d.file}
                        </div>
                      </TableCell>
                      <TableCell className="whitespace-nowrap">
                        {d.version ?? "—"}
                      </TableCell>
                      <TableCell>
                        <span className="font-mono text-xs">
                          {d.md5.slice(0, 8)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {d.chunk_count}
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-xs text-muted-foreground">
                        {d.valid_from ?? "?"}
                        {" → "}
                        {d.valid_until ?? "至今"}
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-xs text-muted-foreground">
                        {formatTime(d.indexed_at)}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="secondary"
                          className={cn("font-normal", status.className)}
                        >
                          {status.label}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          className="text-destructive"
                          onClick={() => setDeleteTarget(d)}
                          aria-label={`删除 ${d.doc_id}`}
                          title="删除"
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </main>

      {/* 同名文件覆盖确认（human-in-loop：409 → 用户确认 → overwrite=true 重发） */}
      <Dialog
        open={conflict !== null}
        onOpenChange={(open) => {
          if (!open) setConflict(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>同名文档已存在</DialogTitle>
            <DialogDescription>
              {conflict?.file.name} 将覆盖已有文档，该文档的索引块会整体重建。
            </DialogDescription>
          </DialogHeader>
          {conflict && (
            <div className="rounded-md border p-3 text-xs">
              <div className="flex justify-between gap-3">
                <span className="text-muted-foreground">doc_id</span>
                <span className="font-mono">{conflict.info.doc_id}</span>
              </div>
              <div className="mt-1 flex justify-between gap-3">
                <span className="text-muted-foreground">标题</span>
                <span>{conflict.info.title}</span>
              </div>
              <div className="mt-1 flex justify-between gap-3">
                <span className="text-muted-foreground">当前版本</span>
                <span>{conflict.info.version ?? "—"}</span>
              </div>
              <div className="mt-1 flex justify-between gap-3">
                <span className="text-muted-foreground">当前块数</span>
                <span>{conflict.info.chunk_count}</span>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConflict(null)}
              disabled={uploading}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={() =>
                conflict && void upload(conflict.file, true)
              }
              disabled={uploading}
            >
              {uploading && <Loader2 className="size-4 animate-spin" />}
              覆盖上传
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 删除确认 */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除文档</DialogTitle>
            <DialogDescription>
              将删除文件并移除其全部索引块，此操作不可撤销。
            </DialogDescription>
          </DialogHeader>
          {deleteTarget && (
            <div className="rounded-md border p-3 text-xs">
              <div className="flex justify-between gap-3">
                <span className="text-muted-foreground">doc_id</span>
                <span className="font-mono">{deleteTarget.doc_id}</span>
              </div>
              <div className="mt-1 flex justify-between gap-3">
                <span className="text-muted-foreground">标题</span>
                <span>{deleteTarget.title}</span>
              </div>
              <div className="mt-1 flex justify-between gap-3">
                <span className="text-muted-foreground">块数</span>
                <span>{deleteTarget.chunk_count}</span>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteTarget(null)}
              disabled={deleting}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={doDelete}
              disabled={deleting}
            >
              {deleting && <Loader2 className="size-4 animate-spin" />}
              确认删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
