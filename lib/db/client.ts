import { Pool, QueryResult } from "pg";

type SqlParam = string | number | boolean | null;
type QueryOptions = {
  queryTimeoutMs?: number;
  statementTimeoutMs?: number;
};

declare global {
  var __dbPool__: Pool | undefined;
}

function readDbConfig() {
  const connectionString = process.env.DATABASE_URL;
  const host = process.env.DB_HOST ?? process.env.POSTGRES_HOST ?? "localhost";
  const port = Number(process.env.DB_PORT ?? process.env.POSTGRES_PORT ?? "5432");
  const user = process.env.DB_USERNAME ?? process.env.POSTGRES_USER ?? "postgres";
  const password = process.env.DB_PASSWORD ?? process.env.POSTGRES_PASSWORD;
  const database = process.env.DB_NAME ?? process.env.POSTGRES_DB ?? "postgres";
  const max = Number(process.env.DB_MAX_CONNS ?? process.env.POSTGRES_MAX_CONNS ?? "25");
  const min = Number(process.env.DB_MIN_CONNS ?? process.env.POSTGRES_MIN_CONNS ?? "5");
  const sslMode = process.env.DB_SSL ?? process.env.POSTGRES_SSL_MODE;
  const sslEnabled = sslMode === "true" || sslMode === "require";
  const rejectUnauthorized =
    process.env.DB_SSL_REJECT_UNAUTHORIZED === "true" ||
    process.env.POSTGRES_SSL_REJECT_UNAUTHORIZED === "true";

  return {
    connectionString,
    host,
    port,
    user,
    password,
    database,
    max,
    min,
    sslEnabled,
    rejectUnauthorized,
  };
}

function getPool(): Pool {
  if (global.__dbPool__) return global.__dbPool__;

  const cfg = readDbConfig();

  if (!cfg.connectionString && !cfg.password) {
    throw new Error(
      "Missing database credentials. Set DATABASE_URL or DB_HOST/DB_PORT/DB_USERNAME/DB_PASSWORD/DB_NAME (or POSTGRES_* variants).",
    );
  }

  const pool = new Pool({
    connectionString: cfg.connectionString,
    host: cfg.connectionString ? undefined : cfg.host,
    port: cfg.connectionString ? undefined : cfg.port,
    user: cfg.connectionString ? undefined : cfg.user,
    password: cfg.connectionString ? undefined : cfg.password,
    database: cfg.connectionString ? undefined : cfg.database,
    max: cfg.max,
    min: cfg.min,
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 5_000,
    ssl: cfg.sslEnabled ? { rejectUnauthorized: cfg.rejectUnauthorized } : false,
  });

  pool.on("error", (err) => {
    console.error("Unexpected DB pool error:", err);
  });

  global.__dbPool__ = pool;
  return pool;
}

async function resetPool(): Promise<void> {
  if (!global.__dbPool__) return;
  try {
    await global.__dbPool__.end();
  } catch {
    // Ignore close errors and recreate on next use.
  } finally {
    global.__dbPool__ = undefined;
  }
}

export async function queryDb<T = Record<string, unknown>>(
  sql: string,
  params: SqlParam[] = [],
  options: QueryOptions = {},
): Promise<QueryResult<T>> {
  const runQuery = async () => {
    const pool = getPool();
    return pool.query<T>({
      text: sql,
      values: params,
      query_timeout: options.queryTimeoutMs ?? 5_000,
      statement_timeout: options.statementTimeoutMs ?? 5_000,
    });
  };

  try {
    return await runQuery();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const isSslMismatch =
      /does not support SSL connections/i.test(message) ||
      /no pg_hba\.conf entry.*ssl/i.test(message);

    if (isSslMismatch) {
      process.env.DB_SSL = "false";
      process.env.POSTGRES_SSL_MODE = "disable";
      await resetPool();
      return runQuery();
    }

    throw error;
  }
}

export async function closeDbPool(): Promise<void> {
  if (!global.__dbPool__) return;
  await global.__dbPool__.end();
  global.__dbPool__ = undefined;
}
