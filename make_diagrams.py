"""Architecture diagrams for the report and slides.   Run:  python make_diagrams.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config as C

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = {"device": ("#e8f3ea", "#2e7d32"), "cloud": ("#e8eef8", "#1f4e79"), "policy": ("#fdf3e1", "#b7791f"),
       "chain": ("#f3e8f6", "#6a3d9a"), "user": ("#f2f2f2", "#666666")}


def box(ax, x, y, w, h, text, kind, bold=False, fs=9):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, dashed=False, color="#333333", off=(0, 0.08)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=color,
                                 linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], label, fontsize=7.5, style="italic", ha="center",
                color="#444444")


# ------------------------------------------------------------------ full system architecture
fig, ax = plt.subplots(figsize=(10, 6.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.6)
ax.axis("off")
ax.text(1.9, 6.45, "Tourist smartphone (edge)", ha="center", fontsize=11, weight="bold", color="#2e7d32")
ax.text(5.6, 6.45, "Cloud services", ha="center", fontsize=11, weight="bold", color="#1f4e79")
ax.text(8.65, 6.45, "Authorities", ha="center", fontsize=11, weight="bold", color="#6a3d9a")
ax.plot([3.75, 3.75], [0.2, 6.3], ls=":", color="#aaaaaa")
ax.plot([7.35, 7.35], [0.2, 6.3], ls=":", color="#aaaaaa")

box(ax, 0.3, 5.45, 3.2, 0.7, "Tourist app: registration,\ndigital-ID QR, GPS stream", "user")
box(ax, 0.3, 4.3, 3.2, 0.8, "Stage 1 · Geo-fencing\nHaversine · hazard zones · route\ndeviation · idle ≥ 15 min", "device")
box(ax, 0.3, 3.15, 3.2, 0.8, "Stage 2a · On-device AI\nONNX logistic regression\n(0.7 KB, ~10 µs)", "device", bold=True)
box(ax, 0.3, 2.0, 3.2, 0.8, "Escalation policy\nconfidence < θc  or\nHIGH / CRITICAL ?", "policy", bold=True)
box(ax, 0.3, 0.55, 3.2, 0.8, "Accept local result\n(no network; offline queue)", "device")

box(ax, 4.1, 4.3, 3.0, 0.8, "Identity service\nissue / verify digital ID", "cloud")
box(ax, 4.1, 2.0, 3.0, 0.8, "Stage 2b · Cloud LLM\nrisk verification (6.8 s)", "cloud", bold=True)
box(ax, 4.1, 0.55, 3.0, 0.8, "Alert dispatcher\nrisk level → action", "cloud")

box(ax, 7.6, 3.6, 2.2, 2.3, "Permissioned\nblockchain (PoA)\n\nDigital-ID registry\nIncident audit log\n\nTourism Dept ·\nCity Police ·\nImmigration", "chain", bold=False)
box(ax, 7.6, 0.55, 2.2, 1.3, "Police / authority\ndashboard &\nverifier app", "chain")

arrow(ax, 1.9, 5.45, 1.9, 5.1, "GPS fixes", off=(0.45, 0))
arrow(ax, 1.9, 4.3, 1.9, 3.95, "event + 5 features", off=(0.8, 0))
arrow(ax, 1.9, 3.15, 1.9, 2.8, "class, confidence", off=(0.75, 0))
arrow(ax, 1.9, 2.0, 1.9, 1.35, "no", off=(0.2, 0))
arrow(ax, 3.5, 2.4, 4.1, 2.4, "yes", off=(0, 0.1))
arrow(ax, 3.5, 0.95, 4.1, 0.95)
arrow(ax, 5.6, 2.0, 5.6, 1.35)
arrow(ax, 3.5, 5.8, 4.4, 5.1, "register", off=(0.15, 0.12))
arrow(ax, 7.1, 4.7, 7.6, 4.7)
ax.text(7.35, 5.25, "REGISTER_ID", fontsize=7.5, style="italic", ha="center", color="#444444")
arrow(ax, 7.1, 0.95, 7.6, 1.0, "notify", off=(0, 0.12))
arrow(ax, 6.6, 1.35, 8.0, 3.6, "INCIDENT tx", off=(0.55, -0.1))
arrow(ax, 8.7, 1.85, 8.7, 3.6, "verify ID", dashed=True, color="#6a3d9a", off=(0.4, 0))
fig.savefig(C.FIG_DIR / "fig0_system_architecture.png")
plt.close(fig)
print("saved fig0_system_architecture.png")

# ------------------------------------------------------------------ blockchain structure
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.6)
ax.axis("off")
labels = [("Block n−1", "REGISTER_ID × 25"), ("Block n", "REVOKE_ID\nINCIDENT"), ("Block n+1", "INCIDENT")]
for i, (t, txs) in enumerate(labels):
    x = 0.3 + i * 3.3
    box(ax, x, 0.5, 2.8, 2.7, "", "chain")
    ax.text(x + 1.4, 2.95, t, ha="center", fontsize=10, weight="bold", color="#6a3d9a")
    ax.text(x + 0.15, 2.55, "prev_hash: SHA-256(prev header)\nmerkle_root: tx hashes\nvalidator: PoA authority\n"
            "timestamp\nsignature: Ed25519", fontsize=7.2, va="top", family="DejaVu Sans Mono")
    box(ax, x + 0.25, 0.65, 2.3, 0.75, f"Signed transactions\n{txs}", "user", fs=7.5)
    if i < 2:
        arrow(ax, x + 2.8, 1.85, x + 3.3, 1.85, "hash", off=(0, 0.12), color="#6a3d9a")
fig.savefig(C.FIG_DIR / "fig10_blockchain_structure.png")
plt.close(fig)
print("saved fig10_blockchain_structure.png")

# ------------------------------------------------------------------ digital-ID sequence
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.4)
ax.axis("off")
actors = [("Tourist app", 1.0), ("Tourism Dept.\n(issuer node)", 3.6), ("Blockchain", 6.2), ("Police\nverifier app", 8.8)]
for name, x in actors:
    ax.text(x, 4.15, name, ha="center", va="center", fontsize=9, weight="bold",
            bbox=dict(boxstyle="round", fc="#f3e8f6", ec="#6a3d9a"))
    ax.plot([x, x], [0.2, 3.8], color="#bbbbbb", lw=1, ls="--")
steps = [(1.0, 3.6, 3.45, "1. KYC: name, passport/Aadhaar, itinerary, dates"),
         (3.6, 6.2, 3.0, "2. REGISTER_ID: tourist_id + hashes only (signed)"),
         (6.2, 6.2, 2.6, "3. sealed in a block (PoA)"),
         (3.6, 1.0, 2.2, "4. signed credential → QR code"),
         (1.0, 8.8, 1.7, "5. officer scans QR"),
         (8.8, 6.2, 1.25, "6. look up record, check signature, hash, revocation, dates"),
         (6.2, 8.8, 0.75, "7. VALID / REJECTED (+ incident history)")]
for x1, x2, y, t in steps:
    if x1 == x2:
        ax.text(x1 + 0.15, y, t, fontsize=8, va="center")
    else:
        arrow(ax, x1, y, x2, y)
        ax.text((x1 + x2) / 2, y + 0.12, t, fontsize=8, ha="center")
fig.savefig(C.FIG_DIR / "fig11_digital_id_flow.png")
plt.close(fig)
print("saved fig11_digital_id_flow.png")

# ------------------------------------------------------------------ deployed application
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.2)
ax.axis("off")
box(ax, 0.2, 2.6, 2.9, 2.3, "Tourist phone (PWA)\n\nGPS · Leaflet map\ngeo.js: geofence gate +\non-device AI model\noffline queue · SOS\ndigital-ID QR", "device")
box(ax, 0.2, 0.3, 2.9, 1.7, "Police dashboard\n\nlive map · incidents\nverify ID (camera QR)\nzones · blockchain", "chain")
box(ax, 3.9, 0.3, 3.0, 4.6, "FastAPI server\n\nREST API + WebSocket\nauth (PBKDF2, signed tokens)\nevent handling & policy\nSQLite database\n\nPermissioned blockchain\n(digital IDs, incidents)", "cloud", bold=False)
box(ax, 7.6, 3.1, 2.2, 1.6, "Cloud LLM\nClaude / OpenAI /\nGemini (API key)", "policy")
box(ax, 7.6, 0.9, 2.2, 1.5, "Rules engine\n(no key / LLM\nunavailable)", "user")
arrow(ax, 3.1, 4.1, 3.9, 4.1, "HTTPS: register,", off=(0, 0.12))
ax.text(3.5, 3.75, "track, events, SOS", fontsize=7.5, style="italic", ha="center", color="#444444")
arrow(ax, 3.9, 3.2, 3.1, 3.2, "decision", off=(0, 0.12))
arrow(ax, 3.1, 1.4, 3.9, 1.4, "REST", off=(0, 0.12))
arrow(ax, 3.9, 0.85, 3.1, 0.85, "live feed (WebSocket)", off=(0, -0.25))
arrow(ax, 6.9, 3.9, 7.6, 3.9, "escalated", off=(0, 0.12))
arrow(ax, 6.9, 1.65, 7.6, 1.65, "fallback", off=(0, 0.12))
fig.savefig(C.FIG_DIR / "fig12_deployed_app.png")
plt.close(fig)
print("saved fig12_deployed_app.png")
