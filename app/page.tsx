"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import { ChatCanvas } from "@/components/chat-canvas";
import { ChatInputBar } from "@/components/chat-input-bar";

export default function Home() {
  const [input, setInput] = useState("");
  const [compactMode, setCompactMode] = useState(false);
  const { messages, sendMessage, status, error, setMessages } = useChat();

  const isSending = status === "submitted" || status === "streaming";

  return (
    <div className="relative min-h-screen bg-linear-to-b from-zinc-50 via-zinc-100 to-zinc-100 dark:from-zinc-950 dark:via-zinc-950 dark:to-zinc-900">
      <div className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-4 py-4 sm:px-6 sm:py-6">
        <header className="mb-4 rounded-2xl border border-zinc-200/80 bg-white/80 p-4 shadow-sm backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
                Modern AI Chat
              </h1>
              <p className="mt-1 text-sm text-zinc-500">
                Ask anything. Answers render in markdown with source links.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setCompactMode((current) => !current)}
              className="rounded-lg border border-zinc-300 px-2.5 py-1.5 text-xs font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
            >
              {compactMode ? "Comfortable" : "Compact"}
            </button>
          </div>
        </header>

        <ChatCanvas
          messages={messages}
          isStreaming={isSending}
          compact={compactMode}
        />

        {error ? (
          <p className="mb-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
            Error: {error.message}
          </p>
        ) : null}

        <form
          onSubmit={(event) => {
            event.preventDefault();
            const prompt = input.trim();
            if (!prompt || isSending) return;
            sendMessage({ text: prompt });
            setInput("");
          }}
        >
          <ChatInputBar
            input={input}
            isSending={isSending}
            hasMessages={messages.length > 0}
            onInputChange={setInput}
            onClear={() => setMessages([])}
            onSend={() => {
              const prompt = input.trim();
              if (!prompt || isSending) return;
              sendMessage({ text: prompt });
              setInput("");
            }}
          />
        </form>
      </div>
    </div>
  );
}
