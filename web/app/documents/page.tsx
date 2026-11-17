"use client";

import { useMemo, useRef, useState } from "react";
import { Database, Loader2, Plus, RefreshCw, Trash2, Upload, X } from "lucide-react";

import { DocStatusBadge, EmptyState, ErrorState, LoadingRows, PageHeader } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, errorMessage } from "@/lib/api";
import { fmtBytes, fmtDate, pageLabel } from "@/lib/format";
import { useAsync, useInterval } from "@/lib/hooks";
import type { Chunk, DocumentRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

const TERMINAL = new Set(["indexed", "failed"]);

function GroupPicker({
  options,
  selected,
  onChange,
}: {
  options: string[];
  selected: string[];
  onChange: (v: string[]) => void;
}) {
  const [custom, setCustom] = useState("");
  const all = Array.from(new Set([...options, ...selected])).sort();
  const toggle = (g: string) => onChange(selected.includes(g) ? selected.filter((x) => x !== g) : [...selected, g]);
  const addCustom = () => {
    const g = custom.trim().toLowerCase().replace(/\s+/g, "-");
    if (g && !selected.includes(g)) onChange([...selected, g]);
    setCustom("");
  };
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="ACL groups">
        {all.map((g) => {
          const on = selected.includes(g);
          return (
            <button
              key={g}
              type="button"
              onClick={() => toggle(g)}
              aria-pressed={on}
              className={cn(
                "rounded-full border px-2.5 py-1 font-mono text-[11px] transition-colors",
                on ? "border-primary bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {g}
            </button>
          );
        })}
      </div>
      <div className="flex gap-2">
        <Input
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addCustom();
            }
          }}
          placeholder="Add a group…"
          aria-label="New ACL group"
          className="h-8 text-xs"
        />
        <Button type="button" variant="outline" size="sm" onClick={addCustom} disabled={!custom.trim()}>
          <Plus /> Add
        </Button>
      </div>
    </div>
  );
}

function UploadCard({ groups, onUploaded }: { groups: string[]; onUploaded: (doc: DocumentRecord) => void }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [docId, setDocId] = useState("");
  const [acl, setAcl] = useState<string[]>(["all-employees"]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !acl.length) return;
    setBusy(true);
    setError(null);
    try {
      const doc = await api.uploadDocument({ file, aclGroups: acl, title: title.trim() || undefined, docId: docId.trim() || undefined });
      onUploaded(doc);
      setFile(null);
      setTitle("");
      setDocId("");
      if (fileRef.current) fileRef.current.value = "";
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-4 w-4" /> Upload document
        </CardTitle>
        <CardDescription>PDFs go through multimodal parsing; Markdown and text are chunked directly.</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="doc-file">File (.pdf, .md, .txt)</Label>
            <Input
              ref={fileRef}
              id="doc-file"
              type="file"
              accept=".pdf,.md,.markdown,.txt"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              required
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="doc-title">Title (optional)</Label>
              <Input id="doc-title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="From the H1 if empty" />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="doc-id">Doc ID (optional)</Label>
              <Input id="doc-id" value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="From the file name" />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label>Readable by (ACL groups)</Label>
            <GroupPicker options={groups} selected={acl} onChange={setAcl} />
            {!acl.length && <p className="text-xs text-destructive">Select at least one group.</p>}
          </div>
          {error ? <ErrorState error={error} compact /> : null}
          <Button type="submit" disabled={!file || !acl.length || busy} className="w-full sm:w-auto">
            {busy ? <Loader2 className="animate-spin" /> : <Upload />} Upload and ingest
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function ChunkViewer({ doc, onClose }: { doc: DocumentRecord; onClose: () => void }) {
  const chunks = useAsync<Chunk[]>(() => api.documentChunks(doc.doc_id), [doc.doc_id, doc.updated_at]);
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3 space-y-0">
        <div className="min-w-0 space-y-1">
          <CardTitle className="truncate">Chunks · {doc.title}</CardTitle>
          <CardDescription>
            {doc.num_chunks} chunks · parser <span className="font-mono">{doc.parser ?? "-"}</span> · section path shows the contextual header used for
            embedding.
          </CardDescription>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close chunk viewer">
          <X />
        </Button>
      </CardHeader>
      <CardContent>
        {chunks.error ? (
          <ErrorState error={chunks.error} onRetry={() => void chunks.reload()} compact />
        ) : chunks.loading && !chunks.data ? (
          <LoadingRows rows={4} />
        ) : !chunks.data?.length ? (
          <EmptyState title="No chunks" description={doc.status === "indexed" ? "The document produced no chunks." : "Chunks appear once ingestion finishes."} />
        ) : (
          <div className="scrollbar-thin max-h-[70vh] space-y-2 overflow-y-auto pr-1">
            {chunks.data.map((c) => (
              <div key={c.chunk_id} className="rounded-lg border p-3">
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <Badge variant="secondary" className="font-mono">
                    #{c.chunk_index}
                  </Badge>
                  <span className="min-w-0 flex-1 truncate text-muted-foreground">{c.section_path.join(" › ") || "Overview"}</span>
                  {pageLabel(c.page_start, c.page_end) && (
                    <Badge variant="outline" className="font-mono">
                      {pageLabel(c.page_start, c.page_end)}
                    </Badge>
                  )}
                  <Badge variant="outline" className="font-mono">
                    {c.token_count} tok
                  </Badge>
                </div>
                <p className="mt-2 whitespace-pre-line text-xs leading-relaxed">{c.text}</p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function DocumentsPage() {
  const docs = useAsync(() => api.documents(), []);
  const groups = useAsync(() => api.aclGroups(), []);
  const [selected, setSelected] = useState<DocumentRecord | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const busy = useMemo(() => (docs.data ?? []).some((d) => !TERMINAL.has(d.status)), [docs.data]);
  useInterval(() => void docs.reload(), 2000, busy);

  const selectedLive = selected ? (docs.data?.find((d) => d.doc_id === selected.doc_id) ?? selected) : null;

  const ingestCorpus = async () => {
    setIngesting(true);
    setActionError(null);
    try {
      const res = await api.ingestCorpus();
      setNotice(`Queued ${res.queued.length} corpus documents for ingestion.`);
      await docs.reload();
    } catch (err) {
      setActionError(errorMessage(err));
    } finally {
      setIngesting(false);
    }
  };

  const remove = async (doc: DocumentRecord) => {
    if (!window.confirm(`Delete "${doc.title}" and all its chunks from the index?`)) return;
    setActionError(null);
    try {
      await api.deleteDocument(doc.doc_id);
      if (selected?.doc_id === doc.doc_id) setSelected(null);
      docs.setData((prev) => (prev ?? []).filter((d) => d.doc_id !== doc.doc_id));
    } catch (err) {
      setActionError(errorMessage(err));
    }
  };

  const totalChunks = (docs.data ?? []).reduce((s, d) => s + (d.num_chunks || 0), 0);

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader
        title="Knowledge Base"
        description="Documents in the active index, with ACL groups enforced at retrieval time. Ingestion runs in the background: parse → chunk → embed → index."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => void docs.reload()}>
              <RefreshCw className={cn(docs.loading && "animate-spin")} /> Refresh
            </Button>
            <Button size="sm" onClick={() => void ingestCorpus()} disabled={ingesting}>
              {ingesting ? <Loader2 className="animate-spin" /> : <Database />} Ingest demo corpus
            </Button>
          </>
        }
      />

      {notice && (
        <div className="mb-4 flex items-center justify-between rounded-lg border border-success/30 bg-success/5 px-3 py-2 text-sm">
          <span>{notice}</span>
          <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss" className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
      {actionError && (
        <div className="mb-4">
          <ErrorState error={new Error(actionError)} compact />
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_400px]">
        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <div className="space-y-1">
                <CardTitle>Documents</CardTitle>
                <CardDescription>
                  {docs.data ? `${docs.data.length} documents · ${totalChunks.toLocaleString()} chunks` : "Loading…"}
                  {busy && " · ingestion in progress"}
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {docs.error && !docs.data ? (
                <ErrorState error={docs.error} onRetry={() => void docs.reload()} />
              ) : !docs.data ? (
                <LoadingRows />
              ) : docs.data.length === 0 ? (
                <EmptyState
                  title="The index is empty"
                  description="Ingest the bundled multi-domain demo corpus (11 documents with ACLs) or upload your own PDF, Markdown or text files."
                  action={
                    <Button size="sm" onClick={() => void ingestCorpus()} disabled={ingesting}>
                      <Database /> Ingest demo corpus
                    </Button>
                  }
                />
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Document</TableHead>
                      <TableHead>ACL groups</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Pages</TableHead>
                      <TableHead className="text-right">Chunks</TableHead>
                      <TableHead>Parser</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead className="sr-only">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {docs.data.map((d) => (
                      <TableRow
                        key={d.doc_id}
                        data-state={selectedLive?.doc_id === d.doc_id ? "selected" : undefined}
                        className="cursor-pointer"
                        onClick={() => setSelected(d)}
                      >
                        <TableCell className="max-w-[260px]">
                          <div className="truncate font-medium" title={d.title}>
                            {d.title}
                          </div>
                          <div className="truncate font-mono text-[11px] text-muted-foreground" title={`${d.filename} · ${fmtBytes(d.size_bytes)}`}>
                            {d.doc_id}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex max-w-[220px] flex-wrap gap-1">
                            {d.acl_groups.map((g) => (
                              <Badge key={g} variant="muted" className="font-mono text-[10px]">
                                {g}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell>
                          <DocStatusBadge status={d.status} />
                          {d.error && (
                            <div className="mt-1 max-w-[200px] truncate text-[11px] text-destructive" title={d.error}>
                              {d.error}
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{d.pages ?? "-"}</TableCell>
                        <TableCell className="text-right tabular-nums">{d.num_chunks}</TableCell>
                        <TableCell className="font-mono text-xs text-muted-foreground">{d.parser ?? "-"}</TableCell>
                        <TableCell className="whitespace-nowrap text-xs text-muted-foreground">{fmtDate(d.updated_at)}</TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => {
                              e.stopPropagation();
                              void remove(d);
                            }}
                            aria-label={`Delete ${d.title}`}
                          >
                            <Trash2 className="text-muted-foreground" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          {selectedLive && <ChunkViewer doc={selectedLive} onClose={() => setSelected(null)} />}
        </div>

        <div className="space-y-6">
          <UploadCard
            groups={groups.data ?? []}
            onUploaded={(doc) => {
              docs.setData((prev) => [...(prev ?? []).filter((d) => d.doc_id !== doc.doc_id), doc]);
              setNotice(`"${doc.title}" queued for ingestion.`);
            }}
          />
          <Card>
            <CardHeader>
              <CardTitle>How access control works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs leading-relaxed text-muted-foreground">
              <p>Every chunk stores its document&apos;s ACL groups. Retrieval runs with a Qdrant filter on the persona&apos;s groups, so restricted chunks are never scored, reranked or sent to the LLM.</p>
              <p>
                Everyone implicitly has <span className="font-mono">public</span>. Try asking about pay bands as <em>Priya (credit-risk)</em> and then as <em>Meera (hr)</em>.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
