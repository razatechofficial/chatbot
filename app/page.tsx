"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";

export default function Home() {
  const [input, setInput] = useState("");
  const { messages, sendMessage, status, error, setMessages } = useChat();

  const isSending = status === "submitted" || status === "streaming";

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-3xl flex-col px-4 py-6">
      <h1 className="mb-4 text-2xl font-semibold">Grok Chat</h1>

      <div className="mb-4 flex-1 space-y-4 overflow-y-auto rounded border p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-zinc-500">Start by asking a question.</p>
        ) : (
          messages.map((message) => (
            <div key={message.id}>
              <p className="mb-1 text-xs font-medium uppercase text-zinc-500">
                {message.role === "user" ? "User" : "Assistant"}
              </p>
              <div className="whitespace-pre-wrap text-sm">
                {message.parts.map((part, idx) =>
                  part.type === "text" ? (
                    <span key={`${message.id}-${idx}`}>{part.text}</span>
                  ) : null,
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {error ? (
        <p className="mb-2 text-sm text-red-600">Error: {error.message}</p>
      ) : null}

      <form
        className="flex items-center gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          const prompt = input.trim();
          if (!prompt || isSending) return;
          sendMessage({ text: prompt });
          setInput("");
        }}
      >
        <input
          className="flex-1 rounded border p-2 text-sm"
          value={input}
          placeholder="Say something..."
          onChange={(event) => setInput(event.currentTarget.value)}
          disabled={isSending}
        />
        <button
          type="submit"
          className="rounded bg-black px-3 py-2 text-sm text-white disabled:opacity-50"
          disabled={isSending || !input.trim()}
        >
          {isSending ? "Sending..." : "Send"}
        </button>
        <button
          type="button"
          className="rounded border px-3 py-2 text-sm"
          onClick={() => setMessages([])}
          disabled={isSending || messages.length === 0}
        >
          Clear
        </button>
      </form>
    </div>
  );
}
