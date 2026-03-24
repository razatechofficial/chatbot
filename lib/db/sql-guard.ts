const WRITE_OR_DDL_PATTERN =
  /\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|comment|vacuum|analyze|refresh|call|do)\b/i;

const MULTI_STATEMENT_PATTERN = /;(?=\s*\S)/;

const SELECT_START_PATTERN = /^\s*(with\b[\s\S]*?\bselect\b|select\b)/i;

export type GuardedSqlResult =
  | {
      ok: true;
      normalizedSql: string;
    }
  | {
      ok: false;
      reason: string;
    };

export type IdentifierValidationResult =
  | {
      ok: true;
    }
  | {
      ok: false;
      reason: string;
    };

export function normalizeReadonlySql(sql: string, maxLimit = 200): GuardedSqlResult {
  const trimmed = sql.trim().replace(/;+$/, "");

  if (!trimmed) {
    return { ok: false, reason: "SQL cannot be empty." };
  }

  if (!SELECT_START_PATTERN.test(trimmed)) {
    return {
      ok: false,
      reason: "Only SELECT queries are allowed.",
    };
  }

  if (WRITE_OR_DDL_PATTERN.test(trimmed)) {
    return {
      ok: false,
      reason: "Write or schema-changing SQL is not allowed.",
    };
  }

  if (MULTI_STATEMENT_PATTERN.test(trimmed)) {
    return {
      ok: false,
      reason: "Multiple SQL statements are not allowed.",
    };
  }

  const hasLimit = /\blimit\b/i.test(trimmed);
  let normalizedSql = trimmed;

  if (!hasLimit) {
    // Wrap query to enforce a safe outer limit without mutating internal syntax.
    normalizedSql = `SELECT * FROM (${normalizedSql}) AS __readonly_subquery LIMIT ${maxLimit}`;
  } else {
    normalizedSql = normalizedSql.replace(
      /\blimit\s+(\d+)\b/i,
      (_, value: string) => {
        const parsed = Number(value);
        const safeLimit = Number.isFinite(parsed) ? Math.min(parsed, maxLimit) : maxLimit;
        return `LIMIT ${safeLimit}`;
      },
    );
  }

  return { ok: true, normalizedSql };
}

export function validateSqlIdentifiers(
  sql: string,
  allowedTables: string[],
  tableColumns: Record<string, string[]>,
): IdentifierValidationResult {
  const normalized = sql.toLowerCase();
  const allowedSet = new Set(allowedTables.map((t) => t.toLowerCase()));

  const tableMatches = [...normalized.matchAll(/\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b/gi)];
  for (const match of tableMatches) {
    const table = match[1]?.toLowerCase();
    if (table && !allowedSet.has(table)) {
      return {
        ok: false,
        reason: `Table '${table}' is not in allowed selected schema.`,
      };
    }
  }

  const qualifiedColumnMatches = [
    ...normalized.matchAll(/\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b/gi),
  ];
  for (const match of qualifiedColumnMatches) {
    const table = match[1]?.toLowerCase();
    const column = match[2]?.toLowerCase();
    if (!table || !column) continue;
    if (!allowedSet.has(table)) {
      return { ok: false, reason: `Table '${table}' is not allowed.` };
    }
    const cols = tableColumns[table] ?? [];
    if (!cols.includes(column)) {
      return {
        ok: false,
        reason: `Column '${table}.${column}' does not exist in selected schema.`,
      };
    }
  }

  return { ok: true };
}
