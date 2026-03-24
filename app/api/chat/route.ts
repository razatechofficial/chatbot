import { generateText, streamText, UIMessage, convertToModelMessages } from "ai";
import { groq } from "@ai-sdk/groq";
import { z } from "zod";
import { queryDb } from "@/lib/db/client";
import {
  buildSelectedSchemaPromptContext,
  buildTableIndex,
  getKeywordCandidateTables,
  getSchemaSnapshot,
} from "@/lib/db/schema-context";
import { normalizeReadonlySql, validateSqlIdentifiers } from "@/lib/db/sql-guard";

type WebSearchResult = {
  title: string;
  url: string;
  snippet: string;
};

type DbToolResult =
  | {
      ok: true;
      context: string;
    }
  | {
      ok: false;
      context: string;
    };

const DbPlanSchema = z.object({
  action: z.enum(["db", "web", "none"]),
  intent: z.string().min(1).default(""),
  webQuery: z.string().nullable().optional(),
  sql: z.string().nullable().optional(),
  params: z.array(z.union([z.string(), z.number(), z.boolean(), z.null()])).default([]),
  selectedTables: z.array(z.string()).optional(),
  reason: z.string().nullable().optional(),
});

type PlannerPath =
  | "deterministic_override"
  | "normal_planner"
  | "planner_error_fallback"
  | "planner_invalid_json_fallback"
  | "planner_shape_fallback"
  | "planner_parse_fallback"
  | "db_fallback_schema_unavailable";

type PlannerDecision = {
  plan: z.infer<typeof DbPlanSchema>;
  plannerPath: PlannerPath;
};

const SqlRepairSchema = z.object({
  sql: z.string().min(1),
  reason: z.string().optional(),
});

function extractJsonObject(text: string): string | null {
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) return null;
  return text.slice(firstBrace, lastBrace + 1);
}

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

type PlannerResult = z.infer<typeof DbPlanSchema>;

function getRouteOverride(userText: string): "web" | "none" | null {
  const normalized = userText.toLowerCase();

  if (
    /\b(weather|forecast|temperature|rain|humidity|wind|news|headlines|search|web|internet|google|wikipedia)\b/.test(
      normalized,
    )
  ) {
    return "web";
  }

  if (/^(hi|hello|hey|thanks|thank you)\b/.test(normalized)) {
    return "none";
  }

  return null;
}

function isLikelyDbIntent(userText: string): boolean {
  const normalized = userText.toLowerCase();
  return /\b(table|tables|database|db|sql|record|records|count|row|rows|user|users|email|registered|register|latest|last)\b/.test(
    normalized,
  );
}

async function generateDbFallbackPlan(
  userText: string,
  modelId: string,
): Promise<PlannerResult | null> {
  try {
    const result = await generateText({
      model: groq(modelId),
      temperature: 0,
      prompt: `User request: "${userText}"

You must return JSON only, and action must be "db".
Generate a best-effort read-only SQL query using common likely tables (users, vendors, vendor_services) when exact schema is not available.
Use only SELECT / WITH ... SELECT and positional params if needed.

Return STRICT JSON only:
{
  "action": "db",
  "intent": string,
  "webQuery": null,
  "sql": string,
  "params": Array<string|number|boolean|null>,
  "selectedTables": Array<string>,
  "reason": string | null
}`,
    });

    const maybeJson = extractJsonObject(result.text);
    if (!maybeJson) return null;
    const parsedRaw = JSON.parse(maybeJson) as Record<string, unknown>;
    const parsed = {
      ...parsedRaw,
      action: "db",
      reason:
        parsedRaw.reason == null
          ? "DB fallback planner used due to missing schema snapshot."
          : typeof parsedRaw.reason === "string"
            ? parsedRaw.reason
            : String(parsedRaw.reason),
      webQuery: undefined,
      sql:
        parsedRaw.sql == null
          ? undefined
          : typeof parsedRaw.sql === "string"
            ? parsedRaw.sql
            : String(parsedRaw.sql),
      selectedTables: Array.isArray(parsedRaw.selectedTables)
        ? parsedRaw.selectedTables.map((t) => String(t).toLowerCase())
        : undefined,
    };
    const validated = DbPlanSchema.safeParse(parsed);
    if (!validated.success || validated.data.action !== "db" || !validated.data.sql) {
      return null;
    }
    return validated.data;
  } catch {
    return null;
  }
}

async function planToolInvocation(
  userText: string,
  modelId: string,
): Promise<PlannerDecision> {
  const override = getRouteOverride(userText);
  if (override) {
    return {
      plan: {
        action: override,
        intent: userText,
        webQuery: override === "web" ? userText : undefined,
        sql: undefined,
        params: [],
        reason: `Routed by deterministic override: ${override}`,
      },
      plannerPath: "deterministic_override",
    };
  }

  let schemaContext = "";
  let selectedTables: string[] = [];
  let schemaUnavailable = false;
  try {
    const snapshot = await getSchemaSnapshot();
    const keywordCandidates = getKeywordCandidateTables(userText, snapshot, 8);
    const tableIndex = buildTableIndex(snapshot, keywordCandidates);

    const selector = await generateText({
      model: groq(modelId),
      temperature: 0,
      prompt: `Select only the relevant tables for this request.
User request: "${userText}"

Allowed candidate tables:
${tableIndex}

Return only a comma-separated list of table names. No explanation.`,
      maxTokens: 40,
    });

    const picked = selector.text
      .split(",")
      .map((t) => t.trim().toLowerCase())
      .filter(Boolean);
    const allowed = new Set(keywordCandidates.map((t) => t.toLowerCase()));
    selectedTables = [...new Set(picked)].filter((t) => allowed.has(t));
    if (selectedTables.length === 0) {
      selectedTables = keywordCandidates;
    }

    schemaContext = buildSelectedSchemaPromptContext(snapshot, selectedTables);
  } catch {
    schemaUnavailable = true;
    schemaContext =
      "Schema snapshot unavailable. If this is a database question, still prefer action='db' and produce best-effort read-only SQL using likely tables (users, vendors, vendor_services).";
  }

  let text: string;
  try {
    const result = await generateText({
      model: groq(modelId),
      temperature: 0,
      prompt: `You are a tool router and SQL planner for a chatbot.
User request: "${userText}"

Available actions:
- "db": use PostgreSQL for data questions requiring database lookup.
- "web": use web search for external/current-world information.
- "none": no tool required.

If action is "db":
- generate read-only SQL only (SELECT / WITH ... SELECT).
- never produce writes or DDL.
- use positional params ($1, $2, ...).
- if schema is insufficient, set action="none" and include reason.

If action is "web":
- include webQuery.

Return STRICT JSON only:
{
  "action": "db" | "web" | "none",
  "intent": string,
  "webQuery": string | null,
  "sql": string | null,
  "params": Array<string|number|boolean|null>,
  "selectedTables": Array<string>,
  "reason": string | null
}

Schema context:
${schemaContext}
`,
    });
    text = result.text;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const isRateLimited =
      /rate limit/i.test(message) || /429/.test(message) || /tokens per day/i.test(message);
    return {
      plan: {
        action: "none",
        intent: "",
        webQuery: undefined,
        sql: undefined,
        params: [],
        reason: isRateLimited
          ? "Planner skipped because model rate limit was reached."
          : "Planner call failed; fallback to direct response.",
      },
      plannerPath: "planner_error_fallback",
    };
  }

  const maybeJson = extractJsonObject(text);
  if (!maybeJson) {
    return {
      plan: {
        action: "none",
        intent: "",
        webQuery: null,
        sql: undefined,
        params: [],
        reason: "Planner returned invalid JSON.",
      },
      plannerPath: "planner_invalid_json_fallback",
    };
  }

  try {
    const parsedRaw = JSON.parse(maybeJson) as Record<string, unknown>;
    const parsed = {
      ...parsedRaw,
      reason:
        parsedRaw.reason == null
          ? undefined
          : typeof parsedRaw.reason === "string"
            ? parsedRaw.reason
            : String(parsedRaw.reason),
      webQuery:
        parsedRaw.webQuery == null
          ? undefined
          : typeof parsedRaw.webQuery === "string"
            ? parsedRaw.webQuery
            : String(parsedRaw.webQuery),
      sql:
        parsedRaw.sql == null
          ? undefined
          : typeof parsedRaw.sql === "string"
            ? parsedRaw.sql
            : String(parsedRaw.sql),
      selectedTables:
        Array.isArray(parsedRaw.selectedTables)
          ? parsedRaw.selectedTables.map((t) => String(t).toLowerCase())
          : selectedTables,
    };

    const result = DbPlanSchema.safeParse(parsed);
    if (!result.success) {
      return {
        plan: {
          action: "none",
          intent: "",
          webQuery: undefined,
          sql: undefined,
          params: [],
          reason: "Planner JSON shape invalid; fallback to direct response.",
        },
        plannerPath: "planner_shape_fallback",
      };
    }

    if (
      result.data.action === "none" &&
      schemaUnavailable &&
      isLikelyDbIntent(userText)
    ) {
      const fallback = await generateDbFallbackPlan(userText, modelId);
      if (fallback) {
        return {
          plan: fallback,
          plannerPath: "db_fallback_schema_unavailable",
        };
      }
    }

    return {
      plan: result.data,
      plannerPath: "normal_planner",
    };
  } catch {
    return {
      plan: {
        action: "none",
        intent: "",
        webQuery: undefined,
        sql: undefined,
        params: [],
        reason: "Planner JSON parse failed; fallback to direct response.",
      },
      plannerPath: "planner_parse_fallback",
    };
  }
}

async function runDbToolIfNeeded(
  plan: PlannerResult,
  userText: string,
  modelId: string,
): Promise<DbToolResult> {
  if (plan.action !== "db") {
    return { ok: true, context: "" };
  }

  try {
    if (!plan.sql) {
      return {
        ok: false,
        context: "Database tool failed: planner did not return SQL.",
      };
    }
    let candidateSql = plan.sql;
    const isVendorEmailRequest =
      /\bvendor/.test(userText.toLowerCase()) && /\bemail\b/.test(userText.toLowerCase());
    const plannerUsedVendorsWithoutUsersJoin =
      /\bfrom\s+vendors\b/i.test(candidateSql) && !/\bjoin\s+users\b/i.test(candidateSql);
    if (isVendorEmailRequest && plannerUsedVendorsWithoutUsersJoin) {
      candidateSql = `SELECT COALESCE(v.name, v.business_name) AS name, u.email
FROM vendors v
LEFT JOIN users u ON u.user_id = v.user_id_fk
WHERE u.email IS NOT NULL`;
    }

    const guarded = normalizeReadonlySql(candidateSql);
    if (!guarded.ok) {
      return {
        ok: false,
        context: `Database tool blocked unsafe SQL: ${guarded.reason}`,
      };
    }

    const snapshot = await getSchemaSnapshot();
    const selectedTables = (plan.selectedTables ?? Object.keys(snapshot.tables)).map((t) => t.toLowerCase());
    const effectiveTables = new Set(selectedTables);
    for (const rel of snapshot.relationships) {
      const source = rel.sourceTable.toLowerCase();
      const target = rel.targetTable.toLowerCase();
      if (effectiveTables.has(source)) effectiveTables.add(target);
      if (effectiveTables.has(target)) effectiveTables.add(source);
    }
    const allowedTables = [...effectiveTables];
    const tableColumns = Object.fromEntries(
      Object.entries(snapshot.tables).map(([table, cols]) => [
        table.toLowerCase(),
        cols.map((c) => c.name.toLowerCase()),
      ]),
    );
    const idCheck = validateSqlIdentifiers(
      guarded.normalizedSql,
      allowedTables,
      tableColumns,
    );
    if (!idCheck.ok) {
      return {
        ok: false,
        context: `Database tool blocked invalid identifiers: ${idCheck.reason}`,
      };
    }

    let result;
    try {
      result = await queryDb<Record<string, unknown>>(
        guarded.normalizedSql,
        plan.params,
        { queryTimeoutMs: 5_000, statementTimeoutMs: 5_000 },
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const isSchemaError =
        /column .* does not exist/i.test(message) ||
        /relation .* does not exist/i.test(message) ||
        /\b42703\b/.test(message) ||
        /\b42p01\b/i.test(message);
      if (!isSchemaError) throw error;

      const schemaContext = buildSelectedSchemaPromptContext(snapshot, allowedTables);
      const repair = await generateText({
        model: groq(modelId),
        temperature: 0,
        prompt: `Fix the SQL query using the available schema only.

User request: "${userText}"
Current SQL: ${guarded.normalizedSql}
Execution error: ${message}

Rules:
- Read-only SQL only (SELECT / WITH ... SELECT)
- Use only listed tables/columns
- Prefer joining related tables if needed (for example vendors -> users for email)
- Return STRICT JSON only: {"sql":"...","reason":"..."}

Schema:
${schemaContext}`,
      });

      const repairJson = extractJsonObject(repair.text);
      if (!repairJson) throw error;
      const repairedRaw = JSON.parse(repairJson) as Record<string, unknown>;
      const repaired = SqlRepairSchema.safeParse({
        sql:
          repairedRaw.sql == null
            ? ""
            : typeof repairedRaw.sql === "string"
              ? repairedRaw.sql
              : String(repairedRaw.sql),
        reason:
          repairedRaw.reason == null
            ? undefined
            : typeof repairedRaw.reason === "string"
              ? repairedRaw.reason
              : String(repairedRaw.reason),
      });
      if (!repaired.success) throw error;

      const repairedGuarded = normalizeReadonlySql(repaired.data.sql);
      if (!repairedGuarded.ok) throw error;
      const repairedIdCheck = validateSqlIdentifiers(
        repairedGuarded.normalizedSql,
        allowedTables,
        tableColumns,
      );
      if (!repairedIdCheck.ok) throw error;

      result = await queryDb<Record<string, unknown>>(
        repairedGuarded.normalizedSql,
        plan.params,
        { queryTimeoutMs: 5_000, statementTimeoutMs: 5_000 },
      );
    }

    const previewRows = result.rows.slice(0, 20);
    return {
      ok: true,
      context: [
        "Database tool result:",
        `Intent: ${plan.intent}`,
        `SQL: ${guarded.normalizedSql}`,
        `Row count: ${result.rowCount}`,
        `Columns: ${result.fields.map((f) => f.name).join(", ") || "n/a"}`,
        `Rows preview JSON: ${JSON.stringify(previewRows)}`,
        "Data Source: PostgreSQL (read-only query result).",
      ].join("\n"),
    };
  } catch (error) {
    console.error("Database tool execution failed:", error);
    return {
      ok: false,
      context:
        "Database tool failed due to query planning or execution error. Answer with available context and clearly mention database lookup was unavailable.",
    };
  }
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
    const lastUserText = getLastUserText(sanitizedMessages);
    const decision = await planToolInvocation(lastUserText, modelId);
    const plan = decision.plan;
    console.info("tool_plan", {
      plannerPath: decision.plannerPath,
      action: plan.action,
      selectedTables: plan.selectedTables ?? [],
      hasSql: Boolean(plan.sql),
      hasWebQuery: Boolean(plan.webQuery),
      plannerReason: plan.reason ?? null,
    });
    const enableWebSearch = plan.action === "web";
    const isDatabaseIntent = plan.action === "db";
    const dbToolResult = await runDbToolIfNeeded(plan, lastUserText, modelId);
    const sourceMode = isDatabaseIntent && dbToolResult.ok
      ? "postgresql"
      : enableWebSearch
        ? "web"
        : "none";

    let webContextBlock = "";
    if (enableWebSearch && lastUserText) {
      const webQuery = plan.webQuery?.trim() || lastUserText;
      const { results, error } = await fetchWebSearchResults(webQuery, 3);
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

    let result;
    try {
      result = streamText({
        model: groq(modelId),
        temperature: 0.2,
        system: `You are a helpful assistant. Never output any function-call syntax such as <function=...>.
Always format your answer in clean markdown (headings, bullet lists, tables, fenced code blocks where appropriate).
${enableWebSearch ? "Use provided web context when relevant. Output format must be:\n1) Short answer paragraph(s)\n2) A heading exactly 'Sources'\n3) Bullet list markdown links only, each exactly: - [Title](https://...)\nNever output plain source names, tag lists, 'links tags', or any non-link source format." : ""}
${webContextBlock ? `\n\n${webContextBlock}` : ""}
${dbToolResult.context ? `\n\n${dbToolResult.context}` : ""}
${plan.reason ? `\n\nPlanner note: ${plan.reason}` : ""}
${isDatabaseIntent ? 'This is a database-intent request. Do not use web assumptions or external claims. If database lookup is unavailable, explicitly say you cannot verify the answer because database access failed and ask user to check credentials/table names.' : ""}
For this response, data source policy is "${sourceMode}".
- If source mode is "postgresql": include exactly one line "Data source: PostgreSQL".
- If source mode is "web": include sources section with links and do NOT claim PostgreSQL.
- If source mode is "none": do NOT claim PostgreSQL or web sources.`,
        messages: modelMessages,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (/rate limit|429|tokens per day/i.test(message)) {
        return Response.json(
          {
            error:
              "Model rate limit reached for now. Please wait a few minutes and retry, or switch to a lower-cost model.",
          },
          { status: 429 },
        );
      }
      console.error("Final generation setup failed:", error);
      return Response.json(
        {
          error:
            "I could not generate a response right now. Please retry in a moment.",
        },
        { status: 503 },
      );
    }

    return result.toUIMessageStreamResponse({
      onError: () => "I hit a temporary provider issue. Please try again.",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/rate limit|429|tokens per day/i.test(message)) {
      return Response.json(
        {
          error:
            "Model rate limit reached for now. Please wait a few minutes and retry, or switch to a lower-cost model.",
        },
        { status: 429 },
      );
    }
    console.error("Chat route failed:", error);
    return Response.json(
      { error: "Failed to process chat request." },
      { status: 500 },
    );
  }
}
