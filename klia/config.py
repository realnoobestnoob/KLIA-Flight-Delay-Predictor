"""Configuration: config/config.yaml for tunables, environment variables for secrets."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Load .env from the project root if it exists. Never overwrites variables already in the environment."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)   # setdefault: shell env wins over .env


_load_dotenv()


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = Path(os.environ.get("KLIA_CONFIG", ROOT / "config" / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Model chosen by the notebook wins over config.yaml.
    choice = artifacts_dir() / "model_choice.json"
    if choice.exists():
        with open(choice, encoding="utf-8") as f:
            c = json.load(f)
        cfg["model"]["name"] = c.get("name", cfg["model"]["name"])
        cfg["model"]["params"] = c.get("params", cfg["model"]["params"])
    return cfg


def artifacts_dir() -> Path:
    p = Path(os.environ.get("KLIA_ARTIFACTS_DIR", ROOT / "artifacts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def database_url() -> str | None:
    """Neon gives 'postgresql://...'; SQLAlchemy wants the driver spelled out."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url