import {
  streamText,
  UIMessage,
  convertToModelMessages,
  tool,
  stepCountIs,
} from "ai";
import { groq } from "@ai-sdk/groq";
import { z } from "zod";

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

    const result = streamText({
      model: groq(modelId),
      messages: await convertToModelMessages(messages),
      stopWhen: stepCountIs(5),
      tools: {
        webSearch: tool({
          description:
            "Search the web for recent information and return concise findings with links.",
          inputSchema: z.object({
            query: z.string().describe("Search query from the user"),
            limit: z.number().min(1).max(5).default(3),
          }),
          execute: async ({ query, limit }) => {
            // MVP stub: deterministic mocked search results.
            return {
              query,
              results: Array.from({ length: limit }).map((_, index) => ({
                title: `Result ${index + 1} for "${query}"`,
                url: `https://example.com/search/${encodeURIComponent(query)}/${index + 1}`,
                snippet:
                  "Mocked search result for Phase 2 tool-calling validation.",
              })),
            };
          },
        }),
      },
    });

    return result.toUIMessageStreamResponse();
  } catch {
    return Response.json(
      { error: "Failed to process chat request." },
      { status: 500 },
    );
  }
}
