"use client";

import type { UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type ChatCanvasProps = {
  messages: UIMessage[];
  isStreaming: boolean;
  compact?: boolean;
};

function sanitizeRenderedText(text: string): string {
  return text
    .replace(/<function=.*?>/gi, "")
    .replace(/<\/function>/gi, "")
    .replace(/```(?:xml|json)?\s*<function[\s\S]*?```/gi, "")
    .trim();
}

function extractSources(text: string): Array<{ label: string; url: string }> {
  const start = text.toLowerCase().indexOf("sources");
  if (start === -1) return [];

  const section = text.slice(start);
  const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
  const links: Array<{ label: string; url: string }> = [];
  const seen = new Set<string>();

  let match = linkRegex.exec(section);
  while (match) {
    const label = match[1]?.trim() || "Source";
    const url = match[2]?.trim();
    if (url && !seen.has(url)) {
      seen.add(url);
      links.push({ label, url });
    }
    match = linkRegex.exec(section);
  }

  return links;
}

type MessageRowProps = {
  message: UIMessage;
  isStreaming: boolean;
  compact: boolean;
};

function MessageRow({ message, isStreaming, compact }: MessageRowProps) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 10);
    return () => clearTimeout(t);
  }, []);

  return (
    <div
      className={`flex items-start gap-3 transition-all duration-300 ${
        message.role === "user" ? "justify-end" : "justify-start"
      } ${visible ? "translate-y-0 opacity-100" : "translate-y-1 opacity-0"}`}
    >
      {message.role !== "user" ? (
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-zinc-200 bg-white text-xs font-semibold text-zinc-600 shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200">
          AI
        </div>
      ) : null}
      <div
        className={`max-w-[92%] rounded-2xl shadow-sm ring-1 sm:max-w-[82%] ${
          compact ? "px-3 py-2" : "px-4 py-3"
        } ${
          message.role === "user"
            ? "rounded-br-md bg-linear-to-b from-indigo-600 to-indigo-700 text-white ring-indigo-400/30"
            : "rounded-bl-md bg-white/95 text-zinc-900 ring-zinc-200/90 dark:bg-zinc-900/95 dark:text-zinc-100 dark:ring-zinc-800"
        }`}
      >
        <p
          className={`mb-2 text-[10px] font-semibold uppercase tracking-wide ${
            message.role === "user" ? "text-indigo-100/95" : "text-zinc-500"
          }`}
        >
          {message.role === "user" ? "You" : "Assistant"}
        </p>
        <div className={`text-sm ${compact ? "leading-5" : "leading-6"}`}>
          {message.parts.map((part, idx) => {
            if (part.type === "text") {
              const safeText = sanitizeRenderedText(part.text);
              const sources = extractSources(safeText);
              const isLastAssistantText =
                isStreaming &&
                message.role === "assistant" &&
                idx === message.parts.length - 1;

              return (
                <div key={`${message.id}-${idx}`} className="space-y-2">
                  <div className="prose prose-zinc max-w-none text-inherit dark:prose-invert prose-p:my-1.5 prose-pre:my-2 prose-pre:overflow-x-auto prose-pre:rounded-lg prose-pre:border prose-pre:border-zinc-200 prose-pre:bg-zinc-50 prose-pre:p-3 dark:prose-pre:border-zinc-700 dark:prose-pre:bg-zinc-950 prose-code:rounded prose-code:bg-zinc-200/70 prose-code:px-1 prose-code:py-0.5 prose-code:before:content-[''] prose-code:after:content-[''] dark:prose-code:bg-zinc-800/90">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ({ ...props }) => (
                          <a
                            {...props}
                            target="_blank"
                            rel="noreferrer noopener"
                            className="text-indigo-600 underline decoration-indigo-400 underline-offset-2 dark:text-indigo-400"
                          />
                        ),
                      }}
                    >
                      {safeText}
                    </ReactMarkdown>
                    {isLastAssistantText ? (
                      <span className="ml-1 inline-block h-4 w-2 animate-pulse rounded-sm bg-zinc-500 align-middle dark:bg-zinc-300" />
                    ) : null}
                  </div>
                  {sources.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {sources.map((source) => (
                        <a
                          key={source.url}
                          href={source.url}
                          target="_blank"
                          rel="noreferrer noopener"
                          className="rounded-full border border-zinc-300/80 bg-white/70 px-2 py-1 text-xs text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
                        >
                          {source.label}
                        </a>
                      ))}
                    </div>
                  ) : null}
                </div>
              );
            }

            if (part.type === "tool-webSearch") {
              return (
                <pre
                  key={`${message.id}-${idx}`}
                  className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-2 text-xs dark:border-zinc-700 dark:bg-zinc-950"
                >
                  {JSON.stringify(part, null, 2)}
                </pre>
              );
            }

            return null;
          })}
        </div>
      </div>
      {message.role === "user" ? (
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-indigo-300/40 bg-indigo-600 text-xs font-semibold text-white shadow-sm">
          You
        </div>
      ) : null}
    </div>
  );
}

export function ChatCanvas({ messages, isStreaming, compact = false }: ChatCanvasProps) {
  const endRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isNearBottom, setIsNearBottom] = useState(true);

  const totalMessages = useMemo(() => messages.length, [messages.length]);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const near = distanceFromBottom < 100;
    setIsNearBottom(near);
  }, []);

  useEffect(() => {
    if (isNearBottom) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [totalMessages, isNearBottom]);

  useEffect(() => {
    if (isNearBottom) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [isStreaming, isNearBottom]);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const near = distanceFromBottom < 100;
    setIsNearBottom(near);
  };

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className={`relative mb-4 flex-1 space-y-5 overflow-y-auto rounded-2xl border border-zinc-200/70 bg-white/70 shadow-sm backdrop-blur [scrollbar-width:thin] dark:border-zinc-800 dark:bg-zinc-950/60 ${
        compact ? "p-3 sm:p-4" : "p-4 sm:p-6"
      }`}
    >
      {messages.length === 0 ? (
        <div className="flex h-full min-h-56 items-center justify-center rounded-xl border border-dashed border-zinc-300/80 bg-zinc-50/70 p-6 text-center dark:border-zinc-700 dark:bg-zinc-900/40">
          <p className="text-sm text-zinc-500">
            Start by asking a question. I can answer directly and add live web
            sources when relevant.
          </p>
        </div>
      ) : (
        messages.map((message) => (
          <MessageRow
            key={message.id}
            message={message}
            isStreaming={isStreaming}
            compact={compact}
          />
        ))
      )}
      {!isNearBottom && totalMessages > 0 ? (
        <button
          type="button"
          onClick={() => {
            endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
          }}
          className="sticky bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white shadow-lg dark:bg-zinc-100 dark:text-zinc-900"
        >
          New messages
        </button>
      ) : null}
      <div ref={endRef} />
    </div>
  );
}
