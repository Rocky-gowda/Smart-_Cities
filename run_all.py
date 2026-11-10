"""End-to-end pipeline: data -> models -> ONNX -> hybrid simulation -> results.

Run:  python run_all.py            (full study, ~10-20 min on a laptop)
      python run_all.py --quick    (smaller grids / fewer repeats, ~3 min, for a quick check)

Outputs
  data/      train.csv, val.csv, test.csv, cuts.json
  models/    *.joblib (all sklearn models), *.onnx (exported), deployed_model.onnx/.json
  results/   results.json (every number), *.csv tables
  figures/   produced afterwards by `python make_figures.py`
"""
import argparse
import json
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config as C
from src import hybrid as H
from src import stats as S
from src.generator import generate, split
from src.models import (LEARNED, balanced_lr, candidates, work_per_inference)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def to_onnx(model, path, X):
    from skl2onnx import to_onnx as _to_onnx
    onx = _to_onnx(model, X[:1].astype(np.float32), options={"zipmap": False},
                   target_opset={"": 17, "ai.onnx.ml": 3})
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())


def onnx_predict(path, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})
    return np.asarray(out[0]).astype(int), np.asarray(out[1])


def bench(path, quick):
    args = [sys.executable, "-m", "src.bench_onnx", str(path)]
    if quick:
        args = [sys.executable, "-c",
                f"from src.bench_onnx import main; main(r'{path}', 100, 1000, 1)"]
    out = subprocess.run(args, capture_output=True, text=True, cwd=C.ROOT, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(quick=False):
    R = {"quick_mode": quick}
    n_boot = 300 if quick else C.N_BOOT
    n_mc = 30 if quick else C.N_MC
    n_mc_ci = 200 if quick else C.N_MC_CI
    n_regen = 3 if quick else C.N_REGEN

    # ------------------------------------------------------------ 1. data
    log("Generating synthetic events")
    df, cuts = generate(C.N_TOTAL, C.SEED)
    tr, va, te = split(df, C.SEED, C.N_TRAIN, C.N_VAL, C.N_TEST)
    for name, d in (("train", tr), ("val", va), ("test", te)):
        d.to_csv(C.DATA_DIR / f"{name}.csv", index=False)
    json.dump({"cuts": cuts.tolist(), "classes": C.CLASSES}, open(C.DATA_DIR / "cuts.json", "w"), indent=2)
    Xtr, Xva, Xte = (d[C.FEATURES].to_numpy(np.float32) for d in (tr, va, te))
    ytr, yva, yte = (d["label"].to_numpy() for d in (tr, va, te))
    R["data"] = dict(n_train=len(tr), n_val=len(va), n_test=len(te),
                     class_mix_train=(np.bincount(ytr) / len(ytr)).round(4).tolist())

    # ------------------------------------------------------------ 2. model comparison
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    fitted, comp = {}, []
    for name, (est, grid) in candidates().items():
        if quick:
            grid = {k: v[:1] for k, v in grid.items()}
        log(f"Training {name}")
        if grid:
            gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=-1).fit(Xtr, ytr)
            model, params = gs.best_estimator_, gs.best_params_
        else:
            model, params = clone(est).fit(Xtr, ytr), {}
        fitted[name] = model
        joblib.dump(model, C.MODEL_DIR / f"{name.replace(' ', '_').replace('.', '')}.joblib")
        pv, pt = model.predict(Xva), model.predict(Xte)
        row = dict(model=name, params={k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items()},
                   val_macro_f1=f1_score(yva, pv, average="macro"))
        m = S.metrics(yte, pt)
        row.update(test_accuracy=m["accuracy"], test_macro_f1=m["macro_f1"],
                   test_critical_recall=m["critical_recall"], **S.bootstrap_ci(yte, pt, n_boot))
        comp.append(row)

    # ONNX export + benchmark for learned models
    log("Exporting to ONNX and benchmarking")
    for row in comp:
        name = row["model"]
        row.update(onnx_size_kb=None, mean_us=None, p95_us=None, p99_us=None, ram_mb=None, work=None)
        if name not in LEARNED:
            continue
        path = C.MODEL_DIR / f"{name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')}.onnx"
        try:
            to_onnx(fitted[name], path, Xtr)
            lab, _ = onnx_predict(path, Xte)
            row["onnx_label_match"] = float((lab == fitted[name].predict(Xte)).mean())
            row.update(bench(path, quick))
            row["onnx_size_kb"] = row.pop("size_kb")
        except Exception as e:  # noqa: BLE001
            row["onnx_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            log(f"  {name}: ONNX export failed ({row['onnx_error']})")
        row["work"] = work_per_inference(name, fitted[name], Xte[:500])
    R["comparison"] = comp

    # selection rule: within 1 pp of best val macro-F1, smallest ONNX size then latency
    learned = [r for r in comp if r["model"] in LEARNED and r["onnx_size_kb"] is not None]
    best = max(r["val_macro_f1"] for r in learned)
    eq = [r for r in learned if r["val_macro_f1"] >= best - 0.01]
    sel = min(eq, key=lambda r: (r["onnx_size_kb"], r["mean_us"]))
    R["selection"] = dict(equivalent=[r["model"] for r in eq], selected=sel["model"],
                          best_val_macro_f1=best)
    log(f"Selected: {sel['model']} (equivalent: {R['selection']['equivalent']})")

    # ------------------------------------------------------------ 3. paired tests
    log("Paired significance tests")
    Xtv, ytv = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
    tests = []
    pairs = [("Gradient boosting", "Random forest"), ("Gradient boosting", "Logistic regression"),
             ("Gradient boosting", "Decision tree"), ("MLP (32-16)", "Gradient boosting")]
    for a, b in pairs:
        p_mc = S.mcnemar_exact(yte, fitted[a].predict(Xte), fitted[b].predict(Xte))
        p_f = None
        if not quick:
            p_f = S.five_by_two_cv_f(lambda: clone(fitted[a]), lambda: clone(fitted[b]), Xtv, ytv, C.SEED)
        tests.append(dict(a=a, b=b, mcnemar_p=p_mc, cv5x2_f_p=p_f))
    R["paired_tests"] = tests

    # ------------------------------------------------------------ 4. regenerated datasets
    log(f"Re-training on {n_regen} regenerated datasets")
    regen = {n: [] for n in LEARNED}
    for k in range(n_regen):
        d2, _ = generate(C.N_TOTAL, 1000 + k)
        a, _, c = split(d2, 1000 + k, C.N_TRAIN, C.N_VAL, C.N_TEST)
        for n in LEARNED:
            m = clone(fitted[n]).fit(a[C.FEATURES].to_numpy(np.float32), a["label"].to_numpy())
            regen[n].append(float((m.predict(c[C.FEATURES].to_numpy(np.float32)) == c["label"].to_numpy()).mean()))
    R["regenerated"] = {n: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), runs=v) for n, v in regen.items()}

    # ------------------------------------------------------------ 5. importance + calibration
    sel_model = fitted[sel["model"]]
    rng = np.random.default_rng(C.SEED)
    base = (sel_model.predict(Xva) == yva).mean()
    groups = {"displacement": [0], "idle duration": [1], "hour of day": [2, 3], "hazard flag": [4]}
    imp = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(10):
            Xp = Xva.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]
            drops.append(base - (sel_model.predict(Xp) == yva).mean())
        imp[g] = float(np.mean(drops))
    R["permutation_importance_pp"] = {k: v * 100 for k, v in imp.items()}
    R["ece_selected"] = S.expected_calibration_error(sel_model.predict_proba(Xva), yva)

    # ------------------------------------------------------------ 6. class imbalance
    log("Class-imbalance strategies")
    lr_C = fitted["Logistic regression"].named_steps["clf"].C
    lr_unw = fitted["Logistic regression"]
    lr_bal = balanced_lr(lr_C).fit(Xtr, ytr)
    joblib.dump(lr_bal, C.MODEL_DIR / "Logistic_regression_balanced.joblib")
    strat = {
        "Unweighted": (lr_unw, None),
        "Balanced weights": (lr_bal, None),
        "Cost-sensitive": (lr_unw, "cost"),
        "Balanced + cost": (lr_bal, "cost"),
    }

    def local(model, rule, X):
        p = model.predict_proba(X)
        return p, (S.min_cost_predict(p, C.UNDER_WEIGHTS) if rule else p.argmax(axis=1))

    imb = []
    for name, (model, rule) in strat.items():
        _, pred = local(model, rule, Xte)
        m = S.metrics(yte, pred)
        imb.append(dict(strategy=name, accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                        recall=m["recall"], **S.bootstrap_ci(yte, pred, n_boot)))
    R["imbalance"] = imb
    R["confusion_balanced"] = confusion_matrix(yte, lr_bal.predict(Xte)).tolist()

    # ------------------------------------------------------------ 7. deployed model
    log("Exporting deployed model")
    dep_path = C.MODEL_DIR / "deployed_model.onnx"
    to_onnx(lr_bal, dep_path, Xtr)
    dep = bench(dep_path, quick)
    local_s = dep["mean_us"] / 1e6
    sc, clf = lr_bal.named_steps["scaler"], lr_bal.named_steps["clf"]
    json.dump(dict(features=C.FEATURES, classes=C.CLASSES, scaler_mean=sc.mean_.tolist(),
                   scaler_scale=sc.scale_.tolist(), coef=clf.coef_.tolist(), intercept=clf.intercept_.tolist(),
                   note="p = softmax(coef @ ((x - mean) / scale) + intercept)"),
              open(C.MODEL_DIR / "deployed_model.json", "w"), indent=2)
    R["deployed"] = dep

    # ------------------------------------------------------------ 8. hybrid simulation
    log("Calibrating cloud proxy")
    sigma = H.calibrate_sigma(tr["latent_det"].to_numpy(), ytr, cuts)
    R["cloud_sigma"] = sigma
    ld_va, ld_te = va["latent_det"].to_numpy(), te["latent_det"].to_numpy()
    rng = np.random.default_rng(1)
    R["cloud_test_accuracy"] = float(np.mean([(H.cloud_proxy_predict(ld_te, cuts, sigma, rng) == yte).mean()
                                              for _ in range(50)]))
    pva, pte = lr_bal.predict_proba(Xva), lr_bal.predict_proba(Xte)

    log("Sweeping escalation thresholds on validation")
    n_sweep = 20 if quick else 50
    sweep_va = {pol: [H.evaluate_point(pva, yva, ld_va, cuts, sigma, t, pol, local_s, n_sweep, 7)
                      for t in C.THETA_GRID] for pol in ("conf", "safety")}
    cloud_va = H.evaluate_point(pva, yva, ld_va, cuts, sigma, 1.01, "conf", local_s, n_sweep, 7)["accuracy"]
    target = cloud_va - C.ACC_TOLERANCE_PP / 100
    lat_first = max((p for p in sweep_va["conf"] if p["escalated"] <= C.LATENCY_FIRST_MAX_ESC),
                    key=lambda p: p["theta"])
    balanced = min((p for p in sweep_va["conf"] if p["accuracy"] >= target), key=lambda p: p["theta"])
    safety = min((p for p in sweep_va["safety"] if p["accuracy"] >= target), key=lambda p: p["theta"])
    R["val_target_accuracy"] = target

    log("Evaluating operating points on test")
    ops = []
    for label, pol, th in [("Cloud-only", "conf", 1.01), ("On-device only", "conf", 0.0),
                           ("Hybrid, latency-first", "conf", lat_first["theta"]),
                           ("Hybrid, balanced", "conf", balanced["theta"]),
                           ("Hybrid, safety-aware", "safety", safety["theta"])]:
        r = H.evaluate_point(pte, yte, ld_te, cuts, sigma, th, pol, local_s, n_mc, 11)
        r.update(H.ci_point(pte, yte, ld_te, cuts, sigma, th, pol, n_mc_ci, 13))
        r["accuracy_indep"] = H.evaluate_point(pte, yte, ld_te, cuts, sigma, th, pol, local_s, n_mc, 11,
                                               cloud="indep")["accuracy"]
        r["config"] = label
        r["reduction_vs_cloud"] = 1 - r["total_latency_s"] / C.CLOUD_ONLY_TOTAL
        ops.append(r)
    R["operating_points"] = ops

    log("Test-set trade-off curves")
    n_curve = 20 if quick else 50
    R["curves_test"] = {pol: [H.evaluate_point(pte, yte, ld_te, cuts, sigma, t, pol, local_s, n_curve, 17)
                              for t in C.THETA_GRID] for pol in ("conf", "safety")}

    # system-level comparison of imbalance strategies (safety-aware, tuned on validation)
    sys_imb = []
    for name, (model, rule) in strat.items():
        p_v, l_v = local(model, rule, Xva)
        p_t, l_t = local(model, rule, Xte)
        pts = [H.evaluate_point(p_v, yva, ld_va, cuts, sigma, t, "safety", local_s, 20, 7, local_pred=l_v)
               for t in C.THETA_GRID]
        ok = [p for p in pts if p["accuracy"] >= target]
        th = min(ok, key=lambda p: p["theta"])["theta"] if ok else 1.0
        r = H.evaluate_point(p_t, yte, ld_te, cuts, sigma, th, "safety", local_s, n_mc, 11, local_pred=l_t)
        sys_imb.append(dict(strategy=name, theta=th, accuracy=r["accuracy"],
                            critical_recall=r["critical_recall"], escalated=r["escalated"]))
    R["imbalance_system"] = sys_imb

    # ------------------------------------------------------------ 9. sensitivity
    log("Sensitivity to cloud accuracy")
    sp = next(o for o in ops if o["config"] == "Hybrid, safety-aware")
    R["sensitivity"] = H.sensitivity_sweep(pte, yte, sp["theta"], "safety",
                                           [round(0.5 + 0.05 * k, 2) for k in range(11)], n_mc, 21)

    # ------------------------------------------------------------ 10. stress tests
    log("Distribution-shift stress tests")
    conds = {"In-distribution": {}, "GPS noise x3": {"gps_log_sd": 3 * C.GEN["gps_log_sd"]},
             "Gazetteer error 30%": {"hazard_flag_error": 0.30}, "Prior shift (CRITICAL 18%)": {"prior": 0.18},
             "Combined": {"gps_log_sd": 3 * C.GEN["gps_log_sd"], "hazard_flag_error": 0.30, "prior": 0.18}}
    stress = []
    for i, (name, kw) in enumerate(conds.items()):
        kw = dict(kw)
        prior = kw.pop("prior", None)
        d, _ = generate(20_000, 5000 + i, cuts=cuts, **kw)
        r2 = np.random.default_rng(5000 + i)
        if prior:
            mix = np.array(C.CLASS_MIX)
            mix = np.append(mix[:3] / mix[:3].sum() * (1 - prior), prior)
        else:
            mix = np.array(C.CLASS_MIX)
        idx = np.concatenate([r2.choice(np.where(d["label"] == c)[0], int(round(C.N_TEST * mix[c])), replace=False)
                              for c in range(4)])
        d = d.iloc[idx]
        X, y, ld = d[C.FEATURES].to_numpy(np.float32), d["label"].to_numpy(), d["latent_det"].to_numpy()
        p = lr_bal.predict_proba(X)
        dm = S.metrics(y, p.argmax(axis=1))
        h = H.evaluate_point(p, y, ld, cuts, sigma, sp["theta"], "safety", local_s, n_mc, 23)
        stress.append(dict(condition=name, device_acc=dm["accuracy"], device_crit=dm["critical_recall"],
                           escalated=h["escalated"], hybrid_acc=h["accuracy"], hybrid_crit=h["critical_recall"]))
    R["stress"] = stress

    # ------------------------------------------------------------ save
    json.dump(R, open(C.RESULT_DIR / "results.json", "w"), indent=2, default=float)
    write_tables(R)
    log("Done. Now run:  python make_figures.py")


def write_tables(R):
    pct = lambda x: round(100 * x, 1)
    ci = lambda a: f"[{pct(a[0])}, {pct(a[1])}]"
    rows = []
    for r in R["comparison"]:
        rows.append({"Model": r["model"], "Val macro-F1 (%)": pct(r["val_macro_f1"]),
                     "Test accuracy (%)": pct(r["test_accuracy"]), "95% CI": ci(r["accuracy_ci"]),
                     "Test macro-F1 (%)": pct(r["test_macro_f1"]),
                     "CRITICAL recall (%)": pct(r["test_critical_recall"]), "CRIT 95% CI": ci(r["critical_recall_ci"]),
                     "ONNX size (KB)": None if r["onnx_size_kb"] is None else round(r["onnx_size_kb"], 1),
                     "Latency mean (us)": None if r["mean_us"] is None else round(r["mean_us"], 1)})
    pd.DataFrame(rows).to_csv(C.RESULT_DIR / "table1_models.csv", index=False)
    pd.DataFrame([{"Strategy": r["strategy"], "Accuracy (%)": pct(r["accuracy"]), "95% CI": ci(r["accuracy_ci"]),
                   "Macro-F1 (%)": pct(r["macro_f1"]), **{f"{c} recall (%)": pct(v) for c, v in zip(C.CLASSES, r["recall"])}}
                  for r in R["imbalance"]]).to_csv(C.RESULT_DIR / "table2_imbalance.csv", index=False)
    pd.DataFrame([{"Model": r["model"], "Mean (us)": r["mean_us"], "p95 (us)": r["p95_us"], "p99 (us)": r["p99_us"],
                   "Size (KB)": r["onnx_size_kb"], "RAM (MB)": r["ram_mb"], "Work/inf.": r["work"]}
                  for r in R["comparison"] if r["model"] in LEARNED]).to_csv(C.RESULT_DIR / "table3_benchmark.csv", index=False)
    pd.DataFrame([{"Configuration": o["config"], "Policy": o["policy"], "theta_c": o["theta"],
                   "Escalated (%)": pct(o["escalated"]), "Accuracy (%)": pct(o["accuracy"]), "95% CI": ci(o["accuracy_ci"]),
                   "CRITICAL recall (%)": pct(o["critical_recall"]), "Acc. indep. cloud (%)": pct(o["accuracy_indep"]),
                   "AI-stage latency (s)": round(o["ai_latency_s"], 2), "Total latency (s)": round(o["total_latency_s"], 2),
                   "Reduction (%)": pct(o["reduction_vs_cloud"])} for o in R["operating_points"]]
                 ).to_csv(C.RESULT_DIR / "table4_operating_points.csv", index=False)
    pd.DataFrame(R["stress"]).to_csv(C.RESULT_DIR / "table5_stress.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
