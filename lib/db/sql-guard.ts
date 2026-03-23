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

  const hasLimit = /\blimit\s+\d+\b/i.test(trimmed);
  let normalizedSql = trimmed;

  if (!hasLimit) {
    normalizedSql = `${normalizedSql}\nLIMIT ${maxLimit}`;
  } else {
    normalizedSql = normalizedSql.replace(/\blimit\s+(\d+)\b/i, (_, value: string) => {
      const parsed = Number(value);
      const safeLimit = Number.isFinite(parsed) ? Math.min(parsed, maxLimit) : maxLimit;
      return `LIMIT ${safeLimit}`;
    });
  }

  return { ok: true, normalizedSql };
}
