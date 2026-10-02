"""Storage interface. Two implementations: Postgres (Neon) for production, local files for demos/tests."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol

import pandas as pd


class Store(Protocol):
    def ensure_schema(self) -> None: ...
    def lock(self) -> Iterator[None]: ...
    def get_watermark(self) -> int: ...
    def fetch_new_rows(self, after_id: int, limit: int) -> pd.DataFrame: ...
    def latest_version(self) -> int | None: ...
    def load_bundle_bytes(self, version: int | None = None) -> tuple[int, bytes] | None: ...
    def commit(self, blob: bytes, meta: dict, watermark: int, run: dict) -> int: ...


def open_store(cfg: dict, csv_path: str | None = None) -> "Store":
    from klia.config import artifacts_dir, database_url
    url = database_url()
    if url and not csv_path:
        from klia.store.postgres import PostgresStore
        return PostgresStore(url, cfg)
    from klia.store.files import FileStore
    return FileStore(artifacts_dir() / "store", csv_path or "data/departures.csv", cfg)
