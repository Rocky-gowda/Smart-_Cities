"""Integration, system and performance tests for the hybrid edge-cloud pipeline."""
import json
import time
from datetime import datetime

import numpy as np
import pytest

from src import config as C
from src import hybrid as H
from src.geofence import CandidateEvent
from src.trajectory import PLANNED_ROUTE, SCENARIOS, ZONES, make_trace


# ---------------------------------------------------------------- escalation policy (integration)
@pytest.mark.integration
def test_confidence_policy_threshold_semantics():
    proba = np.array([[0.9, 0.1, 0, 0], [0.5, 0.5, 0, 0]])
    assert H.escalate_mask(proba, 0.0, "conf").tolist() == [False, False]   # never escalate
    assert H.escalate_mask(proba, 1.01, "conf").tolist() == [True, True]   # always escalate
    assert H.escalate_mask(proba, 0.7, "conf").tolist() == [False, True]


@pytest.mark.integration
def test_safety_policy_always_escalates_severe_predictions():
    proba = np.array([[0, 0, 0.99, 0.01], [0, 0, 0.01, 0.99], [0.99, 0.01, 0, 0]])
    assert H.escalate_mask(proba, 0.0, "safety").tolist() == [True, True, False]


@pytest.mark.integration
def test_cloud_sigma_calibrates_to_reported_accuracy(artifacts):
    import pandas as pd
    tr = pd.read_csv(C.DATA_DIR / "train.csv")
    cuts = json.load(open(C.DATA_DIR / "cuts.json"))["cuts"]
    R = json.load(open(C.RESULT_DIR / "results.json"))
    rng = np.random.default_rng(0)
    acc = np.mean([(H.cloud_proxy_predict(tr["latent_det"].to_numpy(), cuts, R["cloud_sigma"], rng)
                    == tr["label"].to_numpy()).mean() for _ in range(20)])
    assert acc == pytest.approx(C.CLOUD_TARGET_ACC, abs=0.005)


@pytest.mark.integration
def test_latency_model_mean():
    draws = H.cloud_latency_draws(200_000, np.random.default_rng(0))
    assert draws.mean() == pytest.approx(C.CLOUD_LATENCY_MEAN, rel=0.01)
    assert draws.std() / draws.mean() == pytest.approx(C.CLOUD_LATENCY_CV, rel=0.03)


@pytest.mark.integration
def test_more_escalation_means_more_latency(artifacts):
    R = json.load(open(C.RESULT_DIR / "results.json"))
    curve = R["curves_test"]["conf"]
    lat = [p["ai_latency_s"] for p in curve]
    esc = [p["escalated"] for p in curve]
    assert all(b >= a - 1e-9 for a, b in zip(esc, esc[1:]))
    assert lat[-1] > lat[0]


# ---------------------------------------------------------------- system (end-to-end)
@pytest.fixture(scope="module")
def system(artifacts):
    from src.system import HybridRiskSystem
    return HybridRiskSystem()


def _run(system, name, seed=0):
    sc = next(s for s in SCENARIOS if s.name == name)
    return system.run_stream(name, make_trace(sc, seed), PLANNED_ROUTE, ZONES,
                             {"context_z": sc.context_z, "incident": sc.incident})


@pytest.mark.system
def test_normal_walk_produces_no_alert(system):
    assert _run(system, "normal_walk") == []


@pytest.mark.system
def test_cafe_break_is_low_risk_and_local(system):
    alerts = _run(system, "cafe_break")
    assert len(alerts) == 1 and alerts[0].final_class in ("LOW", "MODERATE")


@pytest.mark.system
@pytest.mark.parametrize("name", ["hazard_night", "restricted_area"])
def test_hazard_scenarios_are_escalated_and_severe(system, name):
    for seed in range(5):
        a = _run(system, name, seed)[0]
        assert a.escalated and a.decided_by == "cloud LLM"
        assert a.final_class in ("HIGH", "CRITICAL")


@pytest.mark.system
def test_alert_record_is_complete(system):
    a = _run(system, "hazard_night")[0]
    for k in ("stage1_gate", "stage2a_on_device", "stage2b_cloud", "ai_stage_total", "end_to_end_estimate"):
        assert k in a.timings_s
    assert a.action and a.escalation_reason and a.zones


@pytest.mark.system
def test_offline_mode_keeps_local_decision_and_queues(artifacts):
    from src.system import CloudLLMStub, HybridRiskSystem
    R = json.load(open(C.RESULT_DIR / "results.json"))
    cuts = json.load(open(C.DATA_DIR / "cuts.json"))["cuts"]
    sys_off = HybridRiskSystem(cloud=CloudLLMStub(R["cloud_sigma"], cuts, available=False))
    a = _run(sys_off, "hazard_night")[0]
    assert "offline" in a.decided_by and a.final_class == a.local_class
    assert len(sys_off.offline_queue) == 1


@pytest.mark.system
def test_system_matches_simulation_on_test_set(artifacts):
    """The integrated system reproduces the simulated operating point (within noise)."""
    import pandas as pd
    from src.system import CloudLLMStub, HybridRiskSystem
    R = json.load(open(C.RESULT_DIR / "results.json"))
    cuts = json.load(open(C.DATA_DIR / "cuts.json"))["cuts"]
    sim = next(o for o in R["operating_points"] if o["config"] == "Hybrid, safety-aware")
    sysm = HybridRiskSystem(cloud=CloudLLMStub(R["cloud_sigma"], cuts, seed=5))
    te = pd.read_csv(C.DATA_DIR / "test.csv")
    preds, esc = [], []
    for r in te.itertuples():
        ev = CandidateEvent("b", datetime(2026, 1, 1), 0, 0, r.disp_obs_m, r.idle_min, r.hour, int(r.hazard_flag))
        a = sysm.decide(ev, {"disp_true_m": r.disp_true_m, "hazard_true": r.hazard_true,
                             "context_z": r.context_z, "incident": r.incident})
        preds.append(C.CLASSES.index(a.final_class))
        esc.append(a.escalated)
    acc = (np.array(preds) == te["label"].to_numpy()).mean()
    assert np.mean(esc) == pytest.approx(sim["escalated"], abs=1e-9)
    assert acc == pytest.approx(sim["accuracy"], abs=0.02)


@pytest.mark.system
def test_goal_g2_accuracy_within_3pp_of_cloud(artifacts):
    R = json.load(open(C.RESULT_DIR / "results.json"))
    ops = {o["config"]: o for o in R["operating_points"]}
    assert ops["Hybrid, safety-aware"]["accuracy"] >= ops["Cloud-only"]["accuracy"] - 0.035


@pytest.mark.system
def test_goal_g1_latency_reduced(artifacts):
    R = json.load(open(C.RESULT_DIR / "results.json"))
    ops = {o["config"]: o for o in R["operating_points"]}
    assert ops["Hybrid, safety-aware"]["total_latency_s"] < ops["Cloud-only"]["total_latency_s"]


# ---------------------------------------------------------------- performance
@pytest.mark.performance
def test_on_device_inference_under_one_millisecond(artifacts):
    from src.system import OnDeviceModel
    m = OnDeviceModel()
    x = np.zeros((1, 5), np.float32)
    for _ in range(200):
        m.predict_proba(x)
    t = []
    for _ in range(2000):
        t0 = time.perf_counter()
        m.predict_proba(x)
        t.append(time.perf_counter() - t0)
    assert np.percentile(t, 99) < 1e-3


@pytest.mark.performance
def test_deployed_model_is_tiny(artifacts):
    assert (C.MODEL_DIR / "deployed_model.onnx").stat().st_size < 2048


@pytest.mark.performance
def test_geofence_gate_throughput():
    from src.geofence import GeofenceGate
    sc = next(s for s in SCENARIOS if s.name == "hazard_night")
    fixes = make_trace(sc, 0) * 20
    t0 = time.perf_counter()
    GeofenceGate("t", PLANNED_ROUTE, ZONES).process(fixes)
    per_fix = (time.perf_counter() - t0) / len(fixes)
    assert per_fix < 1e-3   # well under 1 ms per GPS fix
