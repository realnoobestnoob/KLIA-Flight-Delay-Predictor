import time

import pytest
from fastapi.testclient import TestClient

from klia.api.app import create_app
from klia.etl.validate import clean
from klia.model.bundle import new_bundle
from klia.pipeline import apply_rows


class StubCache:
    def __init__(self, bundle):
        self.bundle, self.version, self.loaded_at = bundle, 3, time.time()

    def get(self):
        return self.bundle


class StubWeather:
    def get(self, when):
        return None, "fallback"


@pytest.fixture()
def client(cfg, demo_df):
    b = new_bundle(cfg)
    ok, _ = clean(demo_df, cfg)
    apply_rows(b, ok, cfg)
    with TestClient(create_app(cache=StubCache(b), weather=StubWeather())) as c:
        yield c


BODY = {"airline": "airasia", "destination": "Singapore", "scheduled_departure": "2026-10-02T18:30"}


def test_predict_ok(client):
    r = client.post("/v1/predict", json=BODY)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["delay_probability"] <= 1 and j["model_version"] == 3
    assert j["known_airline"] is True and j["weather_source"] == "fallback"
    assert j["predicted_delayed"] == (j["delay_probability"] >= j["threshold"])


def test_unknown_route_still_answers(client):
    r = client.post("/v1/predict", json={**BODY, "airline": "Brand New Air", "destination": "Atlantis"})
    assert r.status_code == 200 and r.json()["known_airline"] is False


def test_validation_errors(client):
    assert client.post("/v1/predict", json={**BODY, "airline": " "}).status_code == 422
    assert client.post("/v1/predict", json={**BODY, "scheduled_departure": "tomorrow-ish"}).status_code == 422
    assert client.post("/v1/predict", json={"airline": "AirAsia"}).status_code == 422


def test_info_endpoints(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/v1/model").json()["model"] == "sgd_log"
    assert "AIRASIA" in client.get("/v1/options").json()["airlines"]


def test_api_key(cfg, demo_df, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    b = new_bundle(cfg)
    apply_rows(b, clean(demo_df, cfg)[0], cfg)
    with TestClient(create_app(cache=StubCache(b), weather=StubWeather())) as c:
        assert c.post("/v1/predict", json=BODY).status_code == 401
        assert c.post("/v1/predict", json=BODY, headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/healthz").status_code == 200          # health check stays open


def test_503_without_model(cfg):
    with TestClient(create_app(cache=StubCache(None), weather=StubWeather())) as c:
        assert c.post("/v1/predict", json=BODY).status_code == 503
        assert c.get("/healthz").json()["status"] == "no_model"


def test_cache_hot_swaps_to_newer_model(tmp_artifacts, cfg, demo_df, monkeypatch):
    """Serving picks up a newer bundle from the registry without a restart and without retraining."""
    import klia.jobs.update as up
    from klia.api.cache import ModelCache
    from klia.jobs.update import run
    from klia.store.base import open_store
    monkeypatch.setattr(up, "load_config", lambda: cfg)
    csv = tmp_artifacts / "d.csv"
    demo_df.head(1500).to_csv(csv, index=False)
    run(bootstrap=True, csv=str(csv))
    store = open_store(cfg, str(csv))
    cache = ModelCache(store, tmp_artifacts / "cache", refresh_seconds=0)
    cache.startup()
    assert cache.version == 1
    demo_df.to_csv(csv, index=False)
    run(csv=str(csv))
    assert cache.refresh() is True and cache.version == 2 and cache.bundle.model.n_learned == len(demo_df)
    assert cache.refresh() is False                       # nothing newer: no reload

    # registry unreachable -> a fresh process still starts from the local disk copy
    class Down:
        def latest_version(self):
            raise ConnectionError("neon is asleep")
    offline = ModelCache(Down(), tmp_artifacts / "cache", refresh_seconds=0)
    offline.startup()
    assert offline.version == 2
