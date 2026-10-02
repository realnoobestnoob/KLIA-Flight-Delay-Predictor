import copy
import pickle

from klia.etl.validate import clean
from klia.features.state import FeatureState
from klia.pipeline import feature_stream


def test_serving_features_equal_training_features(cfg, demo_df):
    """The key MLOps guarantee: what the API computes from the saved state is exactly what training saw."""
    ok, _ = clean(demo_df, cfg)
    n = 1500
    train_state = FeatureState(cfg)
    X_train, _ = feature_stream(ok.iloc[: n + 1], train_state)       # features recorded while learning row n

    serve_state = FeatureState(cfg)
    feature_stream(ok.iloc[:n], serve_state)                          # the state a saved bundle would hold
    served = serve_state.features(ok.iloc[n].to_dict())               # what the API would compute
    assert served == X_train[n]


def test_features_do_not_mutate_state(cfg, demo_df):
    ok, _ = clean(demo_df, cfg)
    st = FeatureState(cfg)
    feature_stream(ok.iloc[:500], st)
    before = pickle.dumps(st)
    for rec in ok.iloc[500:520].to_dict("records"):
        st.features(rec)
    assert pickle.dumps(st) == before


def test_no_label_leakage(cfg, demo_df):
    """Flipping a row's own label must not change that row's features."""
    ok, _ = clean(demo_df, cfg)
    st = FeatureState(cfg)
    feature_stream(ok.iloc[:800], st)
    rec = ok.iloc[800].to_dict()
    a = st.features(rec)
    flipped = copy.deepcopy(rec)
    flipped["is_delayed"] = 1 - rec["is_delayed"]
    assert st.features(flipped) == a


def test_unknown_keys_fall_back_to_prior(cfg, demo_df):
    ok, _ = clean(demo_df, cfg)
    st = FeatureState(cfg)
    feature_stream(ok.iloc[:500], st)
    rec = ok.iloc[500].to_dict() | {"airline": "NEW AIR", "destination": "ATLANTIS"}
    x = st.features(rec)
    assert abs(x["airline_rate"] - st.base_rate) < 1e-9
    assert abs(x["route_rate"] - x["airline_rate"]) < 1e-9
    assert not st.known_airline("NEW AIR")
