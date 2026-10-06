"""Neon / Postgres store. Creates its own small bookkeeping tables; never touches your data table except to read it."""
from __future__ import annotations

import json
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text

LOCK_KEY = 727001

DDL = [
    """CREATE TABLE IF NOT EXISTS etl_state (
           key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS model_registry (
           version SERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
           meta JSONB NOT NULL, blob BYTEA NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS etl_runs (
           run_id SERIAL PRIMARY KEY, ts TIMESTAMPTZ NOT NULL DEFAULT now(),
           version INT, watermark BIGINT, info JSONB NOT NULL)""",
]


class PostgresStore:
    def __init__(self, url: str, cfg: dict):
        self.engine = create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=0, pool_recycle=240)
        self.table = cfg["data"]["table"]
        self.keep = int(cfg["registry"]["keep_bundles"])

    def ensure_schema(self) -> None:
        with self.engine.begin() as c:
            for stmt in DDL:
                c.execute(text(stmt))

    @contextmanager
    def lock(self):
        """Only one update job at a time. Use Neon's *direct* (non-pooled) connection string for the job."""
        conn = self.engine.connect()
        try:
            got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": LOCK_KEY}).scalar()
            if not got:
                raise RuntimeError("another update job is already running")
            yield
        finally:
            try:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_KEY})
            finally:
                conn.close()

    def get_watermark(self) -> int:
        with self.engine.connect() as c:
            v = c.execute(text("SELECT value FROM etl_state WHERE key='watermark_id'")).scalar()
        return int(v) if v is not None else 0

    def count_new_rows(self, after_id: int) -> int:
        """Count rows beyond the watermark that are ready to train on (have actual_departure)."""
        q = text(f"SELECT COUNT(*) FROM {self.table} WHERE id > :a AND actual_departure IS NOT NULL")
        with self.engine.connect() as c:
            return int(c.execute(q, {"a": after_id}).scalar() or 0)

    def fetch_new_rows(self, after_id: int, limit: int) -> pd.DataFrame:
        q = text(f"SELECT * FROM {self.table} WHERE id > :a AND actual_departure IS NOT NULL ORDER BY id LIMIT :n")
        with self.engine.connect() as c:
            return pd.read_sql(q, c, params={"a": after_id, "n": limit})

    def latest_version(self) -> int | None:
        with self.engine.connect() as c:
            return c.execute(text("SELECT max(version) FROM model_registry")).scalar()

    def load_bundle_bytes(self, version: int | None = None):
        sql = "SELECT version, blob FROM model_registry " + ("WHERE version=:v" if version else "ORDER BY version DESC LIMIT 1")
        with self.engine.connect() as c:
            row = c.execute(text(sql), {"v": version} if version else {}).fetchone()
        return None if row is None else (int(row[0]), bytes(row[1]))

    def commit(self, blob: bytes, meta: dict, watermark: int, run: dict) -> int:
        """Bundle, watermark and run log are written in ONE transaction: all or nothing."""
        with self.engine.begin() as c:
            v = c.execute(text("INSERT INTO model_registry (meta, blob) VALUES (CAST(:m AS JSONB), :b) RETURNING version"),
                          {"m": json.dumps(meta, default=str), "b": blob}).scalar()
            c.execute(text("""INSERT INTO etl_state (key, value) VALUES ('watermark_id', :w)
                              ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()"""),
                      {"w": str(watermark)})
            c.execute(text("INSERT INTO etl_runs (version, watermark, info) VALUES (:v, :w, CAST(:i AS JSONB))"),
                      {"v": v, "w": watermark, "i": json.dumps(run, default=str)})
            c.execute(text("DELETE FROM model_registry WHERE version <= :cut"), {"cut": v - self.keep})
        return int(v)
