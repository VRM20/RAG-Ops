"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

const CITE_PREFIX = "#cite-";

/** Turn bare `[n]` citation markers into links the renderer can swap for chips. */
export function linkifyCitations(text: string): string {
  // Skip markers already followed by "(" (real markdown links) or ":" (reference definitions).
  return text.replace(/\[(\d{1,2})\](?![(:])/g, (_, n: string) => `[${n}](${CITE_PREFIX}${n})`);
}

export function MarkdownAnswer({
  text,
  onCite,
  activeSource,
  maxSource,
  className,
}: {
  text: string;
  onCite?: (n: number) => void;
  activeSource?: number | null;
  /** Citations above this number are rendered as invalid (model cited a non-existent source). */
  maxSource?: number;
  className?: string;
}) {
  return (
    <div className={cn("prose-answer", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith(CITE_PREFIX)) {
              const n = Number(href.slice(CITE_PREFIX.length));
              const invalid = maxSource !== undefined && (n < 1 || n > maxSource);
              return (
                <button
                  type="button"
                  onClick={() => !invalid && onCite?.(n)}
                  title={invalid ? `Source [${n}] does not exist` : `Show source ${n}`}
                  aria-label={invalid ? `Invalid citation ${n}` : `Show source ${n}`}
                  className={cn(
                    "mx-0.5 inline-flex h-4 min-w-4 -translate-y-1 items-center justify-center rounded px-1 align-baseline font-mono text-[10px] font-semibold leading-none transition-colors",
                    invalid
                      ? "cursor-not-allowed bg-destructive/15 text-destructive line-through"
                      : activeSource === n
                        ? "bg-primary text-primary-foreground"
                        : "bg-primary/15 text-primary hover:bg-primary/25",
                  )}
                >
                  {n}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">
                {children}
              </a>
            );
          },
        }}
      >
        {linkifyCitations(text)}
      </ReactMarkdown>
    </div>
  );
}
