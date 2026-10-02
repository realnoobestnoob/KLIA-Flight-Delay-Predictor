"""Connectivity and schema check:  python -m klia.jobs.check"""
from __future__ import annotations

import sys

from sqlalchemy import create_engine, inspect, text

from klia.config import database_url, load_config

REQUIRED = ["id", "date", "scheduled_departure", "actual_departure", "airline", "destination"]
OPTIONAL = ["aircraft", "flight_number", *load_config()["features"]["weather_cols"]]


def main() -> int:
    url = database_url()
    if not url:
        print("FAIL  DATABASE_URL is not set (see README step 3)")
        return 1
    cfg = load_config()
    table = cfg["data"]["table"]
    try:
        eng = create_engine(url, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:
        print(f"FAIL  cannot connect to Neon: {e}")
        return 1
    print("OK    connected to Neon")
    cols = {c["name"]: str(c["type"]) for c in inspect(eng).get_columns(table)}
    if not cols:
        print(f"FAIL  table '{table}' not found (change data.table in config/config.yaml)")
        return 1
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        print(f"FAIL  table '{table}' is missing required columns: {missing}")
        print(f"      columns found: {cols}")
        return 1
    print(f"OK    table '{table}' has all required columns")
    for c in OPTIONAL:
        print(f"      {'found  ' if c in cols else 'missing'} optional column {c}")
    with eng.connect() as c:
        n, lo, hi, mx = c.execute(text(
            f"SELECT count(*), min(id), max(id), count(*) FILTER (WHERE actual_departure IS NOT NULL) FROM {table}")).fetchone()
    print(f"OK    {n} rows, id {lo}..{hi}, {mx} with an actual departure time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
