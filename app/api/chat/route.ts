import { streamText, UIMessage, convertToModelMessages } from "ai";
import { groq } from "@ai-sdk/groq";

type WebSearchResult = {
  title: string;
  url: string;
  snippet: string;
};

function shouldUseWebSearch(messages: UIMessage[]): boolean {
  const lastUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");

  if (!lastUserMessage) return false;

  const text = lastUserMessage.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join(" ")
    .toLowerCase();

  return /\b(search|latest|news|current|today|recent|web|weather|forecast|temperature|rain|humidity)\b/.test(
    text,
  );
}

function getLastUserText(messages: UIMessage[]): string {
  const lastUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");

  if (!lastUserMessage) return "";

  return lastUserMessage.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join(" ")
    .trim();
}

async function fetchWebSearchResults(query: string, limit = 3) {
  const tavilyApiKey = process.env.TAVILY_API_KEY;

  if (!tavilyApiKey) {
    return {
      results: [] as WebSearchResult[],
      error:
        "Missing TAVILY_API_KEY. Add it in client/.env.local to enable real web search.",
    };
  }

  const response = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      api_key: tavilyApiKey,
      query,
      max_results: limit,
      include_answer: false,
      include_images: false,
    }),
  });

  if (!response.ok) {
    return {
      results: [] as WebSearchResult[],
      error: `Web search failed with status ${response.status}.`,
    };
  }

  const data = (await response.json()) as {
    results?: Array<{ title?: string; url?: string; content?: string }>;
  };

  const results: WebSearchResult[] = (data.results ?? [])
    .filter((item) => item.url)
    .slice(0, limit)
    .map((item) => ({
      title: item.title ?? "Untitled",
      url: item.url as string,
      snippet: item.content ?? "",
    }));

  return { results, error: null };
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { messages?: UIMessage[] };
    const messages = body.messages;

    if (!Array.isArray(messages)) {
      return Response.json(
        { error: "Invalid request body. Expected { messages: UIMessage[] }." },
        { status: 400 },
      );
    }

    // AI SDK Groq provider expects GROQ_API_KEY.
    // Support GROK_API_KEY as a fallback to avoid setup confusion.
    if (!process.env.GROQ_API_KEY && process.env.GROK_API_KEY) {
      process.env.GROQ_API_KEY = process.env.GROK_API_KEY;
    }

    if (!process.env.GROQ_API_KEY) {
      return Response.json(
        {
          error:
            "Missing API key. Set GROQ_API_KEY (or GROK_API_KEY) in client/.env.local.",
        },
        { status: 500 },
      );
    }

    const modelId = process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile";

    const modelMessages = await convertToModelMessages(messages);
    const enableWebSearch = shouldUseWebSearch(messages);
    const lastUserText = getLastUserText(messages);

    let webContextBlock = "";
    if (enableWebSearch && lastUserText) {
      const { results, error } = await fetchWebSearchResults(lastUserText, 3);
      const sources = results
        .map(
          (item, index) =>
            `${index + 1}. ${item.title}\nURL: ${item.url}\nSnippet: ${item.snippet}`,
        )
        .join("\n\n");

      webContextBlock = error
        ? `Web search status: ${error}`
        : sources
          ? `Use the following web results as context:\n\n${sources}`
          : "Web search returned no results.";
    }

    const result = streamText({
      model: groq(modelId),
      system: `You are a helpful assistant. Never output raw function-call syntax.
${enableWebSearch ? "For search/weather/news requests, use provided web context and include a 'Sources' section with markdown links." : ""}
${webContextBlock ? `\n\n${webContextBlock}` : ""}`,
      messages: modelMessages,
    });

    return result.toUIMessageStreamResponse({
      onError: () => "I hit a temporary provider issue. Please try again.",
    });
  } catch {
    return Response.json(
      { error: "Failed to process chat request." },
      { status: 500 },
    );
  }
}
