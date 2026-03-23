import { streamText, UIMessage, convertToModelMessages } from "ai";
import { groq } from "@ai-sdk/groq";

type WebSearchResult = {
  title: string;
  url: string;
  snippet: string;
};

function stripFunctionSyntax(text: string): string {
  return text
    .replace(/<function=.*?>/gi, "")
    .replace(/<\/function>/gi, "")
    .replace(/```(?:xml|json)?\s*<function[\s\S]*?```/gi, "")
    .trim();
}

function sanitizeUiMessages(messages: UIMessage[]): UIMessage[] {
  return messages.map((message) => ({
    ...message,
    parts: message.parts.map((part) => {
      if (part.type !== "text") return part;
      return {
        ...part,
        text: stripFunctionSyntax(part.text),
      };
    }),
  }));
}

const AUTHORITATIVE_DOMAINS = [
  "wikipedia.org",
  "reuters.com",
  "apnews.com",
  "bbc.com",
  "gov",
  "edu",
];

function getDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "").toLowerCase();
  } catch {
    return "";
  }
}

function scoreResult(result: WebSearchResult): number {
  const domain = getDomain(result.url);
  if (!domain) return -100;
  const isAuthoritative = AUTHORITATIVE_DOMAINS.some((candidate) =>
    domain.endsWith(candidate),
  );
  return isAuthoritative ? 10 : 0;
}

function improveSourceQuality(results: WebSearchResult[], limit = 3): WebSearchResult[] {
  const byDomain = new Map<string, WebSearchResult>();

  for (const result of results) {
    const domain = getDomain(result.url);
    if (!domain) continue;
    if (!byDomain.has(domain)) {
      byDomain.set(domain, result);
    }
  }

  return [...byDomain.values()]
    .sort((a, b) => scoreResult(b) - scoreResult(a))
    .slice(0, limit);
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

  const rawResults: WebSearchResult[] = (data.results ?? [])
    .filter((item) => item.url)
    .map((item) => ({
      title: item.title ?? "Untitled",
      url: item.url as string,
      snippet: item.content ?? "",
    }));

  const results = improveSourceQuality(rawResults, limit);

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

    const sanitizedMessages = sanitizeUiMessages(messages);
    const modelMessages = await convertToModelMessages(sanitizedMessages);
    const enableWebSearch = true;
    const lastUserText = getLastUserText(sanitizedMessages);

    let webContextBlock = "";
    if (enableWebSearch && lastUserText) {
      const { results, error } = await fetchWebSearchResults(lastUserText, 3);
      const sources = results
        .map(
          (item, index) =>
            `${index + 1}. ${item.title}\nURL: ${item.url}\nSnippet: ${item.snippet}`,
        )
        .join("\n\n");
      const sourceLinkTemplate = results
        .map((item) => `- [${item.title}](${item.url})`)
        .join("\n");

      webContextBlock = error
        ? `Web search status: ${error}\nYou should still answer helpfully from your own knowledge, and clearly note that live search failed.`
        : sources
          ? `Use the following web results as context:\n\n${sources}\n\nWhen adding Sources, only use this exact markdown link format:\n${sourceLinkTemplate}`
          : "Web search returned no results. You should still answer helpfully from your own knowledge and note that no live sources were found.";
    }

    const result = streamText({
      model: groq(modelId),
      temperature: 0.2,
      system: `You are a helpful assistant. Never output any function-call syntax such as <function=...>.
${enableWebSearch ? "Use provided web context when relevant. Output format must be:\n1) Short answer paragraph(s)\n2) A heading exactly 'Sources'\n3) Bullet list markdown links only, each exactly: - [Title](https://...)\nNever output plain source names, tag lists, 'links tags', or any non-link source format." : ""}
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
