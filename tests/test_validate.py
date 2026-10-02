import datetime as dt

import pandas as pd

from klia.etl.validate import clean, norm, parse_time


def test_parse_time_mixed_formats():
    assert parse_time("08:05") == dt.time(8, 5)
    assert parse_time("8:05 PM") == dt.time(20, 5)
    assert parse_time("8:05PM") == dt.time(20, 5)
    assert parse_time("12:10 AM") == dt.time(0, 10)
    assert parse_time("a.m.?") is None
    assert parse_time(None) is None
    assert parse_time(dt.time(9, 30)) == dt.time(9, 30)


def _row(sched, act, **kw):
    base = dict(id=1, date="2026-03-01", scheduled_departure=sched, actual_departure=act,
                airline=" airasia ", destination="Singapore", aircraft=None)
    base.update(kw)
    return base


def test_labels_and_overnight(cfg):
    df = pd.DataFrame([
        _row("10:00", "10:20", id=1),        # 20 min late -> delayed
        _row("10:00", "09:55", id=2),        # 5 min EARLY must stay early, not become +24h
        _row("23:50", "00:30", id=3),        # left after midnight -> 40 min late
        _row("10:00", "10:14", id=4),        # 14 min -> not delayed
        _row("10:00", "10:15", id=5),        # exactly 15 -> delayed
    ])
    ok, bad = clean(df, cfg)
    assert bad.empty
    by_id = ok.set_index("id")
    assert by_id.loc[1, "is_delayed"] == 1
    assert by_id.loc[2, "delay_min"] == -5 and by_id.loc[2, "is_delayed"] == 0
    assert by_id.loc[3, "delay_min"] == 40
    assert by_id.loc[4, "is_delayed"] == 0 and by_id.loc[5, "is_delayed"] == 1
    assert (ok["airline"] == "AIRASIA").all() and (ok["aircraft"] == "UNKNOWN").all()


def test_rejects_bad_rows(cfg):
    df = pd.DataFrame([
        _row("25:99", "10:00", id=1),
        _row("10:00", "garbage", id=2),
        _row("10:00", "10:05", id=3, airline=""),
        _row("10:00", "10:05", id=4, date="not a date"),
        _row("10:00", "10:05", id=5),
    ])
    ok, bad = clean(df, cfg)
    assert list(ok["id"]) == [5]
    assert set(bad["id"]) == {1, 2, 3, 4}
    assert bad["reason"].str.len().min() > 0


def test_norm():
    assert norm("  Air   Asia ") == "AIR ASIA"
    assert norm(None) == ""
