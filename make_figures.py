"""Make every figure from results/results.json.   Run:  python make_figures.py"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C
from src.geofence import CircleZone, GeofenceGate, PolygonZone
from src.trajectory import PLANNED_ROUTE, SCENARIOS, ZONES, make_trace

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
BLUE, RED, GREY = "#1f4e79", "#c0504d", "#7f7f7f"
CLS_COL = ["#9dc3e6", "#2e75b6", "#e0a526", "#c0504d"]
R = json.load(open(C.RESULT_DIR / "results.json"))
P = lambda x: 100 * np.asarray(x)


def save(fig, name):
    fig.savefig(C.FIG_DIR / name)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- 2 model comparison
comp = R["comparison"][::-1]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
y = np.arange(len(comp))
for off, key, ci, col, lab in [(0.15, "test_accuracy", "accuracy_ci", BLUE, "Accuracy"),
                               (-0.15, "test_critical_recall", "critical_recall_ci", RED, "CRITICAL recall")]:
    v = P([r[key] for r in comp])
    lo = P([r[ci][0] for r in comp])
    hi = P([r[ci][1] for r in comp])
    ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt="o", color=col, ms=4, capsize=2, label=lab)
learned = [P(r["test_accuracy"]) for r in comp if r["model"] not in ("Majority class", "Rule-based score")]
ax.axvspan(min(learned), max(learned), color=BLUE, alpha=0.08)
ax.set_yticks(y, [r["model"] for r in comp])
ax.set_xlabel(f"Test-set value (%) with 95% bootstrap CI (n = {R['data']['n_test']:,})")
ax.set_xlim(-2, 100)
ax.legend(loc="lower left", frameon=True)
save(fig, "fig2_model_comparison.png")

# ---------------------------------------------------------------- 3 imbalance
imb = R["imbalance"]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(len(imb))
for k in range(4):
    v = P([r["recall"][k] for r in imb])
    ax.bar(x + (k - 1.5) * 0.19, v, 0.19, color=CLS_COL[k], label=C.CLASSES[k])
for i, r in enumerate(imb):
    ax.hlines(P(r["accuracy"]), i - 0.42, i + 0.42, color="black", lw=2)
ax.plot([], [], color="black", lw=2, label="Accuracy")
ax.set_xticks(x, [r["strategy"] for r in imb])
ax.set_ylabel("Test recall (%)")
ax.set_ylim(0, 105)
ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
save(fig, "fig3_imbalance.png")

# ---------------------------------------------------------------- 4 trade-off
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for pol, col, lab in [("conf", BLUE, "Confidence-only policy"), ("safety", RED, "Safety-aware policy")]:
    c = R["curves_test"][pol]
    ax.plot([p["ai_latency_s"] for p in c], P([p["accuracy"] for p in c]), color=col, lw=2, label=lab)
marks = {"Cloud-only": "s", "On-device only": "D", "Hybrid, latency-first": "^", "Hybrid, balanced": "o",
         "Hybrid, safety-aware": "*"}
for o in R["operating_points"]:
    ax.plot(o["ai_latency_s"], P(o["accuracy"]), marks[o["config"]], color="black" if "only" in o["config"] else
            (RED if "safety" in o["config"] else BLUE), ms=12 if "safety" in o["config"] else 8)
    ax.annotate(f"{o['config']} ({P(o['escalated']):.0f}% esc.)", (o["ai_latency_s"], P(o["accuracy"])),
                textcoords="offset points", xytext=(8, 6) if o["config"] == "On-device only" else (8, -12), fontsize=8)
ax.set_xlabel("Mean AI-stage latency per event (s)  [simulated]")
ax.set_ylabel("System accuracy (%)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", frameon=False)
save(fig, "fig4_tradeoff.png")

# ---------------------------------------------------------------- 5 sensitivity
s = R["sensitivity"]
sp = next(o for o in R["operating_points"] if o["config"] == "Hybrid, safety-aware")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(P([p["cloud_acc"] for p in s]), P([p["accuracy"] for p in s]), "o-", color=BLUE, label="System accuracy")
ax.plot(P([p["cloud_acc"] for p in s]), P([p["critical_recall"] for p in s]), "s-", color=RED, label="System CRITICAL recall")
ax.axvline(P(sp["cloud_acc_on_escalated"]), color="#b8860b", ls=":", lw=1.5)
ax.text(P(sp["cloud_acc_on_escalated"]) + 0.5, 52, f"proxy: {P(sp['cloud_acc_on_escalated']):.1f}%", color="#b8860b", fontsize=8)
ax.axhline(P(R["operating_points"][0]["accuracy"]), color=GREY, ls="--", lw=1)
ax.set_xlabel("Cloud accuracy on the escalated subset (%)")
ax.set_ylabel("%")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
save(fig, "fig5_sensitivity.png")

# ---------------------------------------------------------------- 6 latency breakdown
stages = [("Detection", 0.8), ("DB write", 2.1), ("DB update", 1.7), ("Notification", 1.3)]
cfgs = ["Cloud-only", "Hybrid, safety-aware", "Hybrid, balanced", "Hybrid, latency-first", "On-device only"]
ops = {o["config"]: o for o in R["operating_points"]}
fig, ax = plt.subplots(figsize=(6.5, 3.0))
cols = ["#d9d9d9", "#bfbfbf", "#a6a6a6", "#8c8c8c"]
for i, cfg in enumerate(cfgs):
    left = 0
    ai = ops[cfg]["ai_latency_s"]
    ax.barh(i, ai, color=RED if ai > 0.01 else BLUE, label="AI stage" if i == 0 else None)
    left = ai
    for (n, v), c in zip(stages, cols):
        ax.barh(i, v, left=left, color=c, label=n if i == 0 else None)
        left += v
    ax.text(left + 0.15, i, f"{left:.2f} s", va="center", fontsize=8)
ax.set_yticks(range(len(cfgs)), cfgs)
ax.invert_yaxis()
ax.set_xlabel("Mean end-to-end alert latency (s)")
ax.set_xlim(0, 14.5)
ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.18))
save(fig, "fig6_latency_breakdown.png")

# ---------------------------------------------------------------- 7 confusion matrix
cm = np.array(R["confusion_balanced"])
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]}\n{100 * cmn[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if cmn[i, j] > 0.5 else "black")
ax.set_xticks(range(4), C.CLASSES, fontsize=8)
ax.set_yticks(range(4), C.CLASSES, fontsize=8)
ax.set_xlabel("Predicted (on-device)")
ax.set_ylabel("True")
ax.spines[:].set_visible(True)
save(fig, "fig7_confusion.png")

# ---------------------------------------------------------------- 8 geofence map
fig, ax = plt.subplots(figsize=(6.5, 5.0))
for z in ZONES:
    if isinstance(z, CircleZone):
        dlat = z.radius_m / 111320
        dlon = dlat / np.cos(np.radians(z.lat))
        t = np.linspace(0, 2 * np.pi, 100)
        ax.fill(z.lon + dlon * np.cos(t), z.lat + dlat * np.sin(t), color=RED if z.hazard else "#2e7d32", alpha=0.18)
        ax.text(z.lon, z.lat + dlat * 1.1, z.name, ha="center", fontsize=7)
    elif isinstance(z, PolygonZone):
        v = np.array(z.vertices + [z.vertices[0]])
        ax.fill(v[:, 1], v[:, 0], color=RED, alpha=0.18)
        ax.text(v[:, 1].mean(), v[:, 0].max() + 0.0004, z.name, ha="center", fontsize=7)
r = np.array(PLANNED_ROUTE)
ax.plot(r[:, 1], r[:, 0], "-", color=BLUE, lw=3, label="Planned route")
markers = ["o", "s", "^", "D", "v"]
for sc, mk in zip(SCENARIOS, markers):
    fx = make_trace(sc, 0)
    tr = np.array([[f.lat, f.lon] for f in fx])
    ax.plot(tr[:, 1], tr[:, 0], ".", ms=2, alpha=0.6)
    evs = GeofenceGate(sc.name, PLANNED_ROUTE, ZONES).process(fx)
    for e in evs:
        ax.plot(e.lon, e.lat, mk, color="black", ms=8, mfc="yellow", label=f"Event: {sc.name}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect(1 / np.cos(np.radians(12.97)))
ax.legend(fontsize=7, loc="lower left", frameon=True)
ax.ticklabel_format(useOffset=False)
save(fig, "fig8_geofence_map.png")

# ---------------------------------------------------------------- 9 importance
imp = R["permutation_importance_pp"]
fig, ax = plt.subplots(figsize=(5, 2.4))
k = sorted(imp, key=imp.get)
ax.barh(k, [imp[i] for i in k], color=BLUE)
for i, n in enumerate(k):
    ax.text(imp[n] + 0.5, i, f"{imp[n]:.1f} pp", va="center", fontsize=8)
ax.set_xlabel("Accuracy drop when permuted (pp, validation)")
save(fig, "fig9_feature_importance.png")
