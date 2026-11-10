"""System architecture validation and execution.

Run after run_all.py:   python validate_system.py

1. Architecture validation - every component loads and the stages agree:
   ONNX vs scikit-learn parity, embedded-coefficient (Flutter) parity,
   geofence-feature vs training-feature consistency.
2. Execution - GPS scenarios run end to end through geofence -> model ->
   escalation -> cloud stub -> alert, including a cloud-offline run.
3. Bulk evaluation - all 4,000 test events go through the integrated system;
   the results are compared with the offline simulation.
4. Blockchain digital ID - registers 200 demo tourists, verifies, revokes, rejects
   forged / expired IDs, anchors scenario alerts on the chain and tests tamper detection.
Outputs: results/system_validation.json, results/scenario_alerts.csv,
         data/chain.json (+ demo keys, KYC vault, credentials) for the demo app
"""
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.geofence import CandidateEvent
from src.system import CloudLLMStub, HybridRiskSystem, OnDeviceModel, alert_to_dict
from src.trajectory import PLANNED_ROUTE, SCENARIOS, ZONES, make_trace


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    V = {}
    res = json.load(open(C.RESULT_DIR / "results.json"))
    cuts = json.load(open(C.DATA_DIR / "cuts.json"))["cuts"]
    te = pd.read_csv(C.DATA_DIR / "test.csv")
    X = te[C.FEATURES].to_numpy(np.float32)
    y = te["label"].to_numpy()

    # ------------------------------------------------ 1. architecture validation
    print("1. Architecture validation")
    onnx_model = OnDeviceModel()
    sk = joblib.load(C.MODEL_DIR / "Logistic_regression_balanced.joblib")
    p_onnx = onnx_model.predict_proba(X)
    p_sk = sk.predict_proba(X)
    js = json.load(open(C.MODEL_DIR / "deployed_model.json"))
    z = ((X - np.array(js["scaler_mean"])) / np.array(js["scaler_scale"])) @ np.array(js["coef"]).T + np.array(js["intercept"])
    p_js = softmax(z)
    raw = te.iloc[:200]
    ev_feats = np.vstack([CandidateEvent("t", datetime(2026, 1, 1), 0, 0, r.disp_obs_m, r.idle_min, r.hour,
                                         int(r.hazard_flag)).features() for r in raw.itertuples()])
    V["architecture"] = {
        "onnx_vs_sklearn_label_agreement": float((p_onnx.argmax(1) == p_sk.argmax(1)).mean()),
        "onnx_vs_sklearn_max_prob_diff": float(np.abs(p_onnx - p_sk).max()),
        "embedded_coefficients_vs_onnx_max_prob_diff": float(np.abs(p_js - p_onnx).max()),
        "geofence_features_vs_training_features_max_diff": float(np.abs(ev_feats - raw[C.FEATURES].to_numpy(np.float32)).max()),
        "components": ["Stage 1 geofence gate", "Stage 2a ONNX model", "Escalation policy",
                       "Stage 2b cloud stub", "Alert dispatcher", "Offline queue"],
    }
    for k, v in V["architecture"].items():
        if k != "components":
            print(f"   {k}: {v:.6g}")

    # ------------------------------------------------ 2. scenario execution
    print("2. End-to-end scenario execution")
    system = HybridRiskSystem()
    rows, summary = [], []
    for sc in SCENARIOS:
        n_alerts, classes = 0, []
        for seed in range(20):
            fixes = make_trace(sc, seed)
            ctx = {"context_z": sc.context_z, "incident": sc.incident}
            alerts = system.run_stream(f"{sc.name}-{seed:02d}", fixes, PLANNED_ROUTE, ZONES, ctx)
            n_alerts += len(alerts)
            for a in alerts:
                classes.append(a.final_class)
                d = alert_to_dict(a)
                rows.append({"scenario": sc.name, **{k: v for k, v in d.items() if k not in ("features", "timings_s")},
                             **{f"f_{k}": v for k, v in d["features"].items()},
                             **{f"t_{k}": v for k, v in d["timings_s"].items()}})
        summary.append(dict(scenario=sc.name, description=sc.description, tourists=20, events=n_alerts,
                            class_counts={c: classes.count(c) for c in C.CLASSES}))
        print(f"   {sc.name:16s} events={n_alerts:3d}  {summary[-1]['class_counts']}")
    V["scenarios"] = summary

    offline = HybridRiskSystem(cloud=CloudLLMStub(res["cloud_sigma"], cuts, available=False))
    sc = next(s for s in SCENARIOS if s.name == "hazard_night")
    alerts = offline.run_stream("offline-00", make_trace(sc, 0), PLANNED_ROUTE, ZONES)
    V["offline_mode"] = dict(alerts=len(alerts), decided_by=[a.decided_by for a in alerts],
                             queued_escalations=len(offline.offline_queue))
    print(f"   offline mode: {V['offline_mode']}")
    pd.DataFrame(rows).to_csv(C.RESULT_DIR / "scenario_alerts.csv", index=False)

    # ------------------------------------------------ 3. bulk system evaluation
    print("3. Bulk evaluation on the 4,000 test events")
    system = HybridRiskSystem(cloud=CloudLLMStub(res["cloud_sigma"], cuts, seed=99))
    preds, esc, t_dev, t_ai = [], [], [], []
    t0 = time.perf_counter()
    for r in te.itertuples():
        ev = CandidateEvent("bulk", datetime(2026, 12, 16), 0, 0, r.disp_obs_m, r.idle_min, r.hour, int(r.hazard_flag))
        a = system.decide(ev, {"disp_true_m": r.disp_true_m, "hazard_true": r.hazard_true,
                               "context_z": r.context_z, "incident": r.incident})
        preds.append(C.CLASSES.index(a.final_class))
        esc.append(a.escalated)
        t_dev.append(a.timings_s["stage2a_on_device"])
        t_ai.append(a.timings_s["ai_stage_total"])
    wall = time.perf_counter() - t0
    preds = np.array(preds)
    sim = next(o for o in res["operating_points"] if o["config"] == "Hybrid, safety-aware")
    V["bulk"] = dict(
        events=len(te), system_accuracy=float((preds == y).mean()),
        system_critical_recall=float((preds[y == 3] == 3).mean()), escalated=float(np.mean(esc)),
        on_device_latency_mean_us=float(np.mean(t_dev) * 1e6), on_device_latency_p99_us=float(np.percentile(t_dev, 99) * 1e6),
        ai_stage_latency_mean_s=float(np.mean(t_ai)), end_to_end_estimate_s=float(np.mean(t_ai) + C.FIXED_STAGES),
        simulation_accuracy=sim["accuracy"], simulation_critical_recall=sim["critical_recall"],
        simulation_escalated=sim["escalated"], wall_time_s=wall)
    for k, v in V["bulk"].items():
        print(f"   {k}: {v:.4g}")
    # ------------------------------------------------ 4. blockchain digital ID
    print("4. Blockchain digital ID")
    import copy
    from datetime import date
    from src import id_demo
    from src.blockchain import Credential
    t0 = time.perf_counter()
    bc, reg, creds = id_demo.build(200)
    t_reg = (time.perf_counter() - t0) / len(creds)
    t0 = time.perf_counter()
    ver = [reg.verify(c, on=date.fromisoformat(c.valid_from)) for c in creds]
    t_ver = (time.perf_counter() - t0) / len(creds)
    for c in creds[:5]:
        reg.revoke(c.tourist_id, "demo revocation (lost device)", "City Police", seal=False)
    bc.seal_block()
    revoked_rejected = sum(not reg.verify(c, on=date.fromisoformat(c.valid_from)).valid for c in creds[:5])
    forged = [Credential(**{**json.loads(c.to_json()), "name": "Impostor"}) for c in creds[5:25]]
    forged_rejected = sum(not reg.verify(c, on=date.fromisoformat(c.valid_from)).valid for c in forged)
    expired_rejected = sum(not reg.verify(c, on=date(2027, 6, 1)).valid for c in creds[5:25])
    chain_system = HybridRiskSystem(cloud=CloudLLMStub(res["cloud_sigma"], cuts, seed=3), registry=reg)
    anchored = 0
    for i, sc in enumerate(SCENARIOS):
        for k in range(4):
            cred = creds[10 + 4 * i + k]
            from datetime import datetime as _dt
            fixes = make_trace(sc, k, day=_dt.fromisoformat(cred.valid_from))
            anchored += len(chain_system.run_stream(cred.tourist_id, fixes, PLANNED_ROUTE, ZONES,
                                                    {"context_z": sc.context_z, "incident": sc.incident}, credential=cred))
    intact = bc.verify_chain()
    t0 = time.perf_counter()
    bc.seal_block()
    tampered = copy.deepcopy(bc)
    inc_blk = next(b for b in tampered.chain if any(t.kind == "INCIDENT" for t in b.transactions))
    victim = next(t for t in inc_blk.transactions if t.kind == "INCIDENT")
    victim.payload["risk_level"] = "LOW" if victim.payload["risk_level"] != "LOW" else "CRITICAL"   # rewrite history
    tamper_problems = tampered.verify_chain()
    id_demo.save(bc, reg, creds)
    n_tx = sum(len(b.transactions) for b in bc.chain)
    V["blockchain"] = dict(
        authorities=list(bc.public_keys), blocks=len(bc.chain), transactions=n_tx,
        tourists_registered=len(creds), valid_verifications=sum(v.valid for v in ver),
        revoked_rejected=f"{revoked_rejected}/5", forged_rejected=f"{forged_rejected}/20",
        expired_rejected=f"{expired_rejected}/20", incidents_anchored=anchored,
        chain_intact=not intact, tamper_detected=bool(tamper_problems), tamper_report=tamper_problems[:3],
        register_ms_mean=t_reg * 1000, verify_ms_mean=t_ver * 1000,
        chain_file_kb=id_demo.CHAIN.stat().st_size / 1024,
        personal_data_on_chain=any(c.name in id_demo.CHAIN.read_text() for c in creds[:50]))
    for k, v in V["blockchain"].items():
        print(f"   {k}: {v}")
    json.dump(V, open(C.RESULT_DIR / "system_validation.json", "w"), indent=2)
    print("Saved results/system_validation.json and results/scenario_alerts.csv")


if __name__ == "__main__":
    main()
