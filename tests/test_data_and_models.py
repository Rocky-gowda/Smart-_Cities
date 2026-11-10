"""Unit tests: synthetic data generator, features, statistics and the trained models."""
import json

import joblib
import numpy as np
import pytest

from src import config as C
from src import stats as S
from src.generator import features_from_raw, generate

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def small():
    return generate(5000, 123)


def test_generator_is_deterministic():
    a, _ = generate(500, 7)
    b, _ = generate(500, 7)
    assert a.equals(b)


def test_class_mix_matches_published(small):
    df, _ = small
    mix = np.bincount(df["label"], minlength=4) / len(df)
    assert np.allclose(mix, C.CLASS_MIX, atol=0.01)


def test_idle_respects_stage1_gate(small):
    df, _ = small
    assert df["idle_min"].min() >= C.GEN["idle_min"]


def test_gps_floor_applied(small):
    df, _ = small
    assert df["disp_obs_m"].min() >= C.GEN["gps_floor_m"]


def test_hazard_flag_error_rate(small):
    df, _ = small
    err = (df["hazard_flag"] != df["hazard_true"]).mean()
    assert err == pytest.approx(C.GEN["hazard_flag_error"], abs=0.02)


def test_features_from_raw_matches_generator(small):
    df, _ = small
    r = df.iloc[0]
    x = features_from_raw(r.disp_obs_m, r.idle_min, r.hour, r.hazard_flag)
    assert np.allclose(x[0], r[C.FEATURES].to_numpy(np.float32), atol=1e-5)


def test_cost_rule_prefers_critical_when_uncertain():
    # 30% CRITICAL vs 70% MODERATE: missing CRITICAL costs 10x, so predict a severe class
    assert S.min_cost_predict(np.array([[0.0, 0.7, 0.0, 0.3]]), C.UNDER_WEIGHTS)[0] >= 2


def test_mcnemar_identical_predictions():
    y = np.array([0, 1, 2, 3])
    assert S.mcnemar_exact(y, y, y) == 1.0


def test_ece_perfect_confidence():
    proba = np.eye(4)[[0, 1, 2, 3]]
    assert S.expected_calibration_error(proba, np.array([0, 1, 2, 3])) == pytest.approx(0)


# ---------------------------------------------------------------- trained artefacts
def test_onnx_matches_sklearn(artifacts):
    from src.system import OnDeviceModel
    import pandas as pd
    te = pd.read_csv(C.DATA_DIR / "test.csv")
    X = te[C.FEATURES].to_numpy(np.float32)
    sk = joblib.load(C.MODEL_DIR / "Logistic_regression_balanced.joblib")
    p = OnDeviceModel().predict_proba(X)
    assert (p.argmax(1) == sk.predict(X)).all()
    assert np.abs(p - sk.predict_proba(X)).max() < 1e-4


def test_probabilities_are_valid(artifacts):
    from src.system import OnDeviceModel
    p = OnDeviceModel().predict_proba(np.random.default_rng(0).normal(size=(100, 5)))
    assert np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()


def test_embedded_coefficients_reproduce_model(artifacts):
    from src.system import OnDeviceModel
    js = json.load(open(C.MODEL_DIR / "deployed_model.json"))
    X = np.random.default_rng(1).normal(size=(50, 5)).astype(np.float32)
    z = ((X - js["scaler_mean"]) / np.array(js["scaler_scale"])) @ np.array(js["coef"]).T + js["intercept"]
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert np.abs(p - OnDeviceModel().predict_proba(X)).max() < 1e-4


def test_selected_model_beats_baselines(artifacts):
    R = json.load(open(C.RESULT_DIR / "results.json"))
    acc = {r["model"]: r["test_accuracy"] for r in R["comparison"]}
    assert acc["Logistic regression"] > acc["Rule-based score"] > acc["Majority class"]


def test_balanced_weights_raise_critical_recall(artifacts):
    R = json.load(open(C.RESULT_DIR / "results.json"))
    imb = {r["strategy"]: r for r in R["imbalance"]}
    assert imb["Balanced weights"]["recall"][3] > imb["Unweighted"]["recall"][3] + 0.15
