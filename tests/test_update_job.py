import pandas as pd

from klia.jobs.update import run
from klia.model.bundle import Bundle
from klia.store.base import open_store


def _setup(tmp_artifacts, cfg, monkeypatch):
    import klia.config as kc
    import klia.jobs.update as up
    monkeypatch.setattr(up, "load_config", lambda: cfg)
    kc.load_config.cache_clear()


def test_incremental_equals_full_replay(tmp_artifacts, cfg, demo_df, monkeypatch):
    _setup(tmp_artifacts, cfg, monkeypatch)
    csv = tmp_artifacts / "d.csv"

    # A: everything at once
    demo_df.to_csv(csv, index=False)
    a = run(bootstrap=True, csv=str(csv))
    assert a["status"] == "ok" and a["rows_learned"] == len(demo_df)
    full = Bundle.loads(open_store(cfg, str(csv)).load_bundle_bytes()[1])

    # B: first 2000 rows, then the remaining 1000 incrementally
    import shutil
    shutil.rmtree(tmp_artifacts / "art")
    demo_df.head(2000).to_csv(csv, index=False)
    run(bootstrap=True, csv=str(csv))
    demo_df.to_csv(csv, index=False)
    b = run(csv=str(csv))
    assert b["rows_in"] == 1000                                   # only NEW rows were read
    inc = Bundle.loads(open_store(cfg, str(csv)).load_bundle_bytes()[1])

    assert inc.model.n_learned == full.model.n_learned == len(demo_df)
    assert inc.state.n == full.state.n and inc.state.s == full.state.s
    probe = {"sched_dt": pd.Timestamp("2026-01-05 18:30").to_pydatetime(), "airline": "AIRASIA", "destination": "SINGAPORE"}
    assert abs(inc.model.predict_proba(inc.state.features(probe)) - full.model.predict_proba(full.state.features(probe))) < 1e-9


def test_rerun_without_new_rows_is_a_noop(tmp_artifacts, cfg, demo_df, monkeypatch):
    _setup(tmp_artifacts, cfg, monkeypatch)
    csv = tmp_artifacts / "d.csv"
    demo_df.head(1500).to_csv(csv, index=False)
    assert run(bootstrap=True, csv=str(csv))["status"] == "ok"
    store = open_store(cfg, str(csv))
    v = store.latest_version()
    assert run(csv=str(csv))["status"] == "no_new_rows"
    assert store.latest_version() == v and store.get_watermark() == 1500


def test_bad_rows_are_rejected_not_fatal(tmp_artifacts, cfg, demo_df, monkeypatch):
    _setup(tmp_artifacts, cfg, monkeypatch)
    df = demo_df.head(1000).copy()
    df.loc[10, "actual_departure"] = "??"
    df.loc[11, "airline"] = None
    csv = tmp_artifacts / "d.csv"
    df.to_csv(csv, index=False)
    r = run(bootstrap=True, csv=str(csv))
    assert r["rows_rejected"] == 2 and r["rows_learned"] == 998


def test_old_bundles_are_pruned(tmp_artifacts, cfg, demo_df, monkeypatch):
    _setup(tmp_artifacts, cfg, monkeypatch)
    csv = tmp_artifacts / "d.csv"
    for n in (600, 900, 1200, 1500, 1800):
        demo_df.head(n).to_csv(csv, index=False)
        run(bootstrap=(n == 600), csv=str(csv))
    store = open_store(cfg, str(csv))
    assert store.latest_version() == 5
    assert len(list((tmp_artifacts / "art" / "store" / "bundles").glob("*.gz"))) == cfg["registry"]["keep_bundles"]
