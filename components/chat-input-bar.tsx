"use client";

type ChatInputBarProps = {
  input: string;
  isSending: boolean;
  hasMessages: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onClear: () => void;
};

export function ChatInputBar({
  input,
  isSending,
  hasMessages,
  onInputChange,
  onSend,
  onClear,
}: ChatInputBarProps) {
  return (
    <div className="sticky bottom-0 rounded-2xl border border-zinc-200/80 bg-white/90 p-2 shadow-lg backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/90">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <input
          className="h-11 w-full rounded-xl border border-zinc-300 bg-white px-3 text-sm outline-none ring-indigo-500 transition placeholder:text-zinc-400 focus:ring-2 dark:border-zinc-700 dark:bg-zinc-900"
          value={input}
          placeholder="Ask anything..."
          onChange={(event) => onInputChange(event.currentTarget.value)}
          disabled={isSending}
        />
        <div className="flex items-center gap-2">
          <button
            type="submit"
            className="h-11 rounded-xl bg-indigo-600 px-4 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isSending || !input.trim()}
            onClick={(event) => {
              event.preventDefault();
              onSend();
            }}
          >
            {isSending ? "Thinking..." : "Send"}
          </button>
          <button
            type="button"
            className="h-11 rounded-xl border border-zinc-300 px-4 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
            onClick={onClear}
            disabled={isSending || !hasMessages}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
