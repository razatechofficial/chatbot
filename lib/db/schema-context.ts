import { queryDb } from "@/lib/db/client";

type TableColumn = {
  tableName: string;
  columnName: string;
  dataType: string;
};

type ForeignKey = {
  sourceTable: string;
  sourceColumn: string;
  targetTable: string;
  targetColumn: string;
};

type SchemaSnapshot = {
  generatedAt: string;
  tables: Record<string, Array<{ name: string; type: string }>>;
  relationships: ForeignKey[];
};

const KEYWORD_TABLE_MAP: Record<string, string[]> = {
  user: ["users", "vendors"],
  users: ["users", "vendors"],
  email: ["users"],
  vendor: ["vendors", "vendor_services"],
  vendors: ["vendors", "vendor_services"],
  service: ["vendor_services", "vendors"],
  services: ["vendor_services", "vendors"],
  price: ["vendor_services"],
  rating: ["vendors"],
  city: ["vendors"],
  state: ["vendors"],
  active: ["users", "vendors", "vendor_services"],
  latest: ["users", "vendors", "vendor_services"],
  last: ["users", "vendors", "vendor_services"],
  created: ["users", "vendors", "vendor_services"],
  signup: ["users"],
  register: ["users"],
};

const DEFAULT_CANDIDATE_TABLES = ["users", "vendors", "vendor_services"];

const CACHE_TTL_MS = 5 * 60 * 1000;
let cache: { value: SchemaSnapshot; expiresAt: number } | null = null;

async function loadColumns(): Promise<TableColumn[]> {
  const sql = `
    SELECT
      c.table_name AS "tableName",
      c.column_name AS "columnName",
      c.data_type AS "dataType"
    FROM information_schema.columns c
    WHERE c.table_schema = 'public'
    ORDER BY c.table_name, c.ordinal_position;
  `;
  const res = await queryDb<TableColumn>(sql);
  return res.rows;
}

async function loadForeignKeys(): Promise<ForeignKey[]> {
  const sql = `
    SELECT
      tc.table_name AS "sourceTable",
      kcu.column_name AS "sourceColumn",
      ccu.table_name AS "targetTable",
      ccu.column_name AS "targetColumn"
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema = 'public'
    ORDER BY tc.table_name, kcu.column_name;
  `;
  const res = await queryDb<ForeignKey>(sql);
  return res.rows;
}

function buildSnapshot(columns: TableColumn[], relationships: ForeignKey[]): SchemaSnapshot {
  const tables: SchemaSnapshot["tables"] = {};

  for (const col of columns) {
    if (!tables[col.tableName]) tables[col.tableName] = [];
    tables[col.tableName].push({
      name: col.columnName,
      type: col.dataType,
    });
  }

  return {
    generatedAt: new Date().toISOString(),
    tables,
    relationships,
  };
}

export async function getSchemaSnapshot(forceRefresh = false): Promise<SchemaSnapshot> {
  if (!forceRefresh && cache && cache.expiresAt > Date.now()) {
    return cache.value;
  }

  const [columns, relationships] = await Promise.all([loadColumns(), loadForeignKeys()]);
  const snapshot = buildSnapshot(columns, relationships);

  cache = {
    value: snapshot,
    expiresAt: Date.now() + CACHE_TTL_MS,
  };

  return snapshot;
}

export function getKeywordCandidateTables(
  userText: string,
  snapshot: SchemaSnapshot,
  maxTables = 8,
): string[] {
  const normalized = userText.toLowerCase();
  const selected = new Set<string>();
  const availableTables = new Set(Object.keys(snapshot.tables));

  for (const [keyword, tables] of Object.entries(KEYWORD_TABLE_MAP)) {
    if (normalized.includes(keyword)) {
      for (const table of tables) {
        if (availableTables.has(table)) selected.add(table);
      }
    }
  }

  if (selected.size === 0) {
    for (const table of DEFAULT_CANDIDATE_TABLES) {
      if (availableTables.has(table)) selected.add(table);
    }
  }

  if (selected.size === 0) {
    for (const table of Object.keys(snapshot.tables).slice(0, maxTables)) {
      selected.add(table);
    }
  }

  return [...selected].slice(0, maxTables);
}

export function buildTableIndex(
  snapshot: SchemaSnapshot,
  tables: string[],
): string {
  return tables
    .map((table) => {
      const cols = (snapshot.tables[table] ?? [])
        .slice(0, 12)
        .map((col) => col.name)
        .join(", ");
      return `${table} — columns: ${cols}`;
    })
    .join("\n");
}

export function buildSelectedSchemaPromptContext(
  snapshot: SchemaSnapshot,
  selectedTables: string[],
): string {
  const selected = selectedTables.filter((t) => snapshot.tables[t]);
  const tableLines = selected.map((tableName) => {
    const cols = snapshot.tables[tableName]
      .map((c) => `${c.name}:${c.type}`)
      .join(", ");
    return `- ${tableName}(${cols})`;
  });

  const relationLines = snapshot.relationships
    .filter(
      (rel) =>
        selected.includes(rel.sourceTable) || selected.includes(rel.targetTable),
    )
    .slice(0, 20)
    .map(
      (rel) =>
        `- ${rel.sourceTable}.${rel.sourceColumn} -> ${rel.targetTable}.${rel.targetColumn}`,
    );

  return [
    `Schema snapshot generated at: ${snapshot.generatedAt}`,
    "Selected tables:",
    ...tableLines,
    relationLines.length ? "Relationships:" : "",
    ...relationLines,
  ]
    .filter(Boolean)
    .join("\n");
}

export function buildSchemaPromptContext(
  snapshot: SchemaSnapshot,
  userText: string,
  maxTables = 8,
): string {
  const normalizedText = userText.toLowerCase();
  const rankedTables = Object.keys(snapshot.tables)
    .map((tableName) => {
      const tableScore = normalizedText.includes(tableName.toLowerCase()) ? 3 : 0;
      const columnScore = snapshot.tables[tableName].some((col) =>
        normalizedText.includes(col.name.toLowerCase()),
      )
        ? 1
        : 0;
      return { tableName, score: tableScore + columnScore };
    })
    .sort((a, b) => b.score - a.score || a.tableName.localeCompare(b.tableName))
    .slice(0, maxTables)
    .map((entry) => entry.tableName);

  const tableLines = rankedTables.map((tableName) => {
    const cols = snapshot.tables[tableName]
      .map((c) => `${c.name}:${c.type}`)
      .join(", ");
    return `- ${tableName}(${cols})`;
  });

  const relationLines = snapshot.relationships
    .filter(
      (rel) =>
        rankedTables.includes(rel.sourceTable) || rankedTables.includes(rel.targetTable),
    )
    .slice(0, 20)
    .map(
      (rel) =>
        `- ${rel.sourceTable}.${rel.sourceColumn} -> ${rel.targetTable}.${rel.targetColumn}`,
    );

  return [
    `Schema snapshot generated at: ${snapshot.generatedAt}`,
    "Relevant tables:",
    ...tableLines,
    relationLines.length ? "Relationships:" : "",
    ...relationLines,
  ]
    .filter(Boolean)
    .join("\n");
}
