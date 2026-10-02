import pytest

from klia.config import load_config
from klia.demo import make


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["model"] = {**c["model"], "name": "hat", "params": {}, "warmup_rows": 200}
    return c


@pytest.fixture(scope="session")
def demo_df():
    return make(rows=3000, seed=3)


@pytest.fixture()
def tmp_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIA_ARTIFACTS_DIR", str(tmp_path / "art"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return tmp_path
