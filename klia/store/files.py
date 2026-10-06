"""Local-file store: same behaviour as Postgres, no database needed. Used for demos, CI and tests."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


class FileStore:
    def __init__(self, root: Path, csv_path: str, cfg: dict):
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.keep = int(cfg["registry"]["keep_bundles"])

    def ensure_schema(self) -> None:
        (self.root / "bundles").mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        self.ensure_schema()
        lk = self.root / "update.lock"
        if lk.exists():
            raise RuntimeError(f"another update job is running (delete {lk} if it crashed)")
        lk.write_text("1")
        try:
            yield
        finally:
            lk.unlink(missing_ok=True)

    def get_watermark(self) -> int:
        p = self.root / "watermark.json"
        return int(json.loads(p.read_text())["watermark_id"]) if p.exists() else 0

    def count_new_rows(self, after_id: int) -> int:
        if not self.csv_path.exists():
            return 0
        df = pd.read_csv(self.csv_path, usecols=["id", "actual_departure"])
        return int(((df["id"] > after_id) & df["actual_departure"].notna()).sum())

    def fetch_new_rows(self, after_id: int, limit: int) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"{self.csv_path} not found")
        df = pd.read_csv(self.csv_path)
        df = df[(df["id"] > after_id) & df["actual_departure"].notna()].sort_values("id")
        return df.head(limit).reset_index(drop=True)

    def _versions(self) -> list[int]:
        d = self.root / "bundles"
        return sorted(int(p.stem.split("_")[1]) for p in d.glob("bundle_*.gz")) if d.exists() else []

    def latest_version(self) -> int | None:
        v = self._versions()
        return v[-1] if v else None

    def load_bundle_bytes(self, version: int | None = None):
        v = version or self.latest_version()
        if v is None:
            return None
        return v, (self.root / "bundles" / f"bundle_{v}.gz").read_bytes()

    def commit(self, blob: bytes, meta: dict, watermark: int, run: dict) -> int:
        self.ensure_schema()
        v = (self.latest_version() or 0) + 1
        (self.root / "bundles" / f"bundle_{v}.gz").write_bytes(blob)
        (self.root / "watermark.json").write_text(json.dumps({"watermark_id": watermark}))
        with open(self.root / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"version": v, "watermark": watermark, **run}, default=str) + "\n")
        for old in [x for x in self._versions() if x <= v - self.keep]:
            (self.root / "bundles" / f"bundle_{old}.gz").unlink(missing_ok=True)
        return v
