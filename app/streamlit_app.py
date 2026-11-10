"""Streamlit demo: Hybrid Edge-Cloud Tourist Safety Monitoring.

Run from the project folder:   streamlit run app/streamlit_app.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.geofence import CandidateEvent, CircleZone, PolygonZone
from src.system import CloudLLMStub, HybridRiskSystem, OnDeviceModel, SEVERITY_ACTION
from src.trajectory import PLANNED_ROUTE, SCENARIOS, ZONES, make_trace

st.set_page_config(page_title="Smart Tourist Safety Monitoring", page_icon="🛡️", layout="wide")

CLASS_COLORS = {"LOW": "#2e7d32", "MODERATE": "#f9a825", "HIGH": "#ef6c00", "CRITICAL": "#c62828"}

if not (C.MODEL_DIR / "deployed_model.onnx").exists():
    st.error("Trained model not found. Run `python run_all.py` first, then reload this page.")
    st.stop()


@st.cache_resource
def load():
    res = json.load(open(C.RESULT_DIR / "results.json"))
    cuts = json.load(open(C.DATA_DIR / "cuts.json"))["cuts"]
    return res, cuts, OnDeviceModel()


RES, CUTS, MODEL = load()


def chain_state():
    """The demo blockchain lives in the session so registrations and alerts persist while you click."""
    if "bc_chain" not in st.session_state:
        from src import id_demo
        if id_demo.CHAIN.exists():
            bc, reg, creds = id_demo.load()
        else:
            bc, reg, creds = id_demo.build(50)
        st.session_state.bc_chain, st.session_state.bc_reg, st.session_state.bc_creds = bc, reg, creds
    return st.session_state.bc_chain, st.session_state.bc_reg, st.session_state.bc_creds
SAFETY = next(o for o in RES["operating_points"] if o["config"] == "Hybrid, safety-aware")


def badge(cls):
    return f"<span style='background:{CLASS_COLORS[cls]};color:white;padding:4px 12px;border-radius:12px;font-weight:600'>{cls}</span>"


st.sidebar.title("🛡️ Smart Tourist Safety")
st.sidebar.caption("Research & analytics dashboard · AI risk classification · Geo-fencing · Blockchain digital ID")
page = st.sidebar.radio("Go to", ["Live risk check", "Geofence simulation", "Digital ID (blockchain)",
                                  "Results dashboard", "System validation & tests", "About"])
st.sidebar.divider()
policy = st.sidebar.selectbox("Escalation policy", ["safety", "conf"],
                              format_func=lambda p: "Safety-aware" if p == "safety" else "Confidence-only")
theta = st.sidebar.slider("Confidence threshold θc", 0.0, 1.0, float(SAFETY["theta"]), 0.025,
                          help="Events whose on-device confidence is below θc are sent to the cloud LLM.")
cloud_online = st.sidebar.toggle("Cloud reachable", value=True)

# =====================================================================================
if page == "Live risk check":
    st.title("Live risk check")
    st.write("Enter one post-gate event. The phone classifies it on-device in microseconds, then the "
             "escalation policy decides whether the cloud LLM must verify it.")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Event (what the phone sees)")
        disp = st.number_input("Displacement from planned route (m)", 0.0, 20000.0, 450.0, 10.0)
        idle = st.number_input("Idle duration (min)", 15.0, 600.0, 35.0, 1.0)
        hour = st.slider("Hour of day", 0.0, 23.99, 22.5, 0.25)
        hazard = st.checkbox("Inside a cached hazard zone", value=True)
        with st.expander("Cloud-only context (the phone cannot see this)"):
            ctx_z = st.slider("Context score (recent reports, location text)", -3.0, 3.0, 0.0, 0.1)
            incident = st.checkbox("Incident reported nearby", value=False)
        seed = st.number_input("Cloud random seed", 0, 9999, 0)

    ev = CandidateEvent("demo", datetime.now(), 12.9716, 77.5946, disp, idle, hour, int(hazard))
    system = HybridRiskSystem(theta=theta, policy=policy, model=MODEL,
                              cloud=CloudLLMStub(RES["cloud_sigma"], CUTS, seed=int(seed), available=cloud_online))
    a = system.decide(ev, {"context_z": ctx_z, "incident": int(incident)})
    proba = MODEL.predict_proba(ev.features())[0]

    with c2:
        st.subheader("Decision")
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"**On-device**<br>{badge(a.local_class)}<br>confidence {a.local_confidence:.2f}", unsafe_allow_html=True)
        m2.markdown(f"**Escalated?**<br><h3 style='margin:0'>{'Yes ☁️' if a.escalated else 'No 📱'}</h3>", unsafe_allow_html=True)
        m3.markdown(f"**Final**<br>{badge(a.final_class)}<br>by {a.decided_by}", unsafe_allow_html=True)
        st.info(f"**Action:** {a.action}  \n**Why:** {a.escalation_reason}")
        import altair as alt
        dfp = pd.DataFrame({"class": C.CLASSES, "probability": proba})
        st.altair_chart(alt.Chart(dfp).mark_bar().encode(
            x=alt.X("class", sort=C.CLASSES, title=None), y=alt.Y("probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("class", sort=C.CLASSES, legend=None,
                            scale=alt.Scale(domain=C.CLASSES, range=[CLASS_COLORS[c] for c in C.CLASSES]))
        ).properties(height=220), width="stretch")
        lt = a.timings_s
        st.markdown("**Latency**")
        l1, l2, l3 = st.columns(3)
        l1.metric("On-device inference", f"{lt['stage2a_on_device'] * 1e6:.0f} µs")
        l2.metric("AI stage", f"{lt['ai_stage_total']:.2f} s", f"{lt['ai_stage_total'] - C.CLOUD_LATENCY_MEAN:+.2f} s vs cloud-only",
                  delta_color="inverse")
        l3.metric("End-to-end (est.)", f"{lt['end_to_end_estimate']:.2f} s", f"{lt['end_to_end_estimate'] - C.CLOUD_ONLY_TOTAL:+.2f} s",
                  delta_color="inverse")
        if not cloud_online and a.escalated:
            st.warning("Cloud offline: the on-device result is final and the escalation request is queued.")

# =====================================================================================
elif page == "Geofence simulation":
    st.title("Geofence simulation (end to end)")
    st.write("A simulated tourist walks a planned route in central Bangalore. Stage 1 checks every GPS fix "
             "against the geofences and raises an event after 15 minutes idle; Stage 2 classifies it.")
    c1, c2 = st.columns([1, 2.2])
    with c1:
        sc = st.selectbox("Scenario", SCENARIOS, format_func=lambda s: s.name.replace("_", " ").title())
        st.caption(sc.description)
        seed = st.number_input("Trace seed", 0, 999, 0)
        bc, reg, creds = chain_state()
        cred = st.selectbox("Tourist (blockchain digital ID)", creds[5:45],
                            format_func=lambda c: f"{c.tourist_id} · {c.name} ({c.nationality})")
        record = st.checkbox("Anchor alerts on the blockchain", value=False,
                             help="Adds an INCIDENT transaction and seals a new block.")
        fixes = make_trace(sc, int(seed), day=datetime.fromisoformat(cred.valid_from))
        system = HybridRiskSystem(theta=theta, policy=policy, model=MODEL,
                                  cloud=CloudLLMStub(RES["cloud_sigma"], CUTS, seed=int(seed), available=cloud_online),
                                  registry=reg if record else None)
        if not record:
            v = reg.verify(cred, on=fixes[0].t.date())
            st.caption(f"Digital ID check: {'✅ valid' if v.valid else '❌ ' + v.reason} (block {v.block})")
        alerts = system.run_stream(sc.name, fixes, PLANNED_ROUTE, ZONES,
                                   {"context_z": sc.context_z, "incident": sc.incident},
                                   credential=cred if record else None)
        st.metric("GPS fixes processed", len(fixes))
        st.metric("Events raised by Stage 1", len(alerts))
        for a in alerts:
            st.markdown(f"**{a.time[11:16]}** · local {badge(a.local_class)} → final {badge(a.final_class)}",
                        unsafe_allow_html=True)
            st.write(f"Zones: {', '.join(a.zones) or 'none'} · {a.features['displacement_m']:.0f} m off route · "
                     f"idle {a.features['idle_min']:.0f} min")
            st.write(f"Escalated: **{a.escalated}** ({a.escalation_reason})")
            st.write(f"Action: **{a.action}**")
            if a.digital_id:
                st.write(f"Digital ID: {'✅ valid' if a.digital_id['valid'] else '❌ ' + a.digital_id['reason']} · "
                         f"anchored as tx `{a.chain_record[:16]}…` in block {len(bc.chain) - 1}")
        if not alerts:
            st.success("No event: the tourist kept moving, so nothing reached Stage 2.")

    with c2:
        from streamlit_folium import st_folium
        from leaflet_map import build_map
        m = build_map(fixes, PLANNED_ROUTE, ZONES, alerts)
        st_folium(m, height=580, use_container_width=True, returned_objects=[], key=f"map-{sc.name}-{seed}")
        st.caption("Leaflet + OpenStreetMap · press ▶ (bottom left) to replay the tourist's movement · "
                   "blue line: planned route · red: hazard zones · green: safe zone · "
                   "warning pins: alerts (click for details)")

# =====================================================================================
elif page == "Digital ID (blockchain)":
    from datetime import date, timedelta
    from src.blockchain import Credential, credential_qr_png
    bc, reg, creds = chain_state()
    st.title("Blockchain-based digital tourist ID")
    st.write("A permissioned proof-of-authority blockchain run by the Tourism Department, City Police and "
             "Immigration Office. Only hashes go on-chain; personal data stays in the issuer's off-chain vault.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Blocks", len(bc.chain))
    k2.metric("Transactions", sum(len(b.transactions) for b in bc.chain))
    k3.metric("Registered tourists", sum(1 for _ in bc.transactions("REGISTER_ID")))
    k4.metric("Incidents anchored", sum(1 for _ in bc.transactions("INCIDENT")))
    t1, t2, t3, t4 = st.tabs(["Register tourist", "Verify ID", "Chain explorer", "Tamper test"])
    with t1:
        with st.form("reg"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Full name", "Asha Verma")
            nat = c2.selectbox("Nationality", ["IN", "DE", "US", "GB", "JP", "FR", "AE", "BR"])
            doc_type = c1.selectbox("Document", ["Aadhaar", "Passport"])
            doc = c2.text_input("Document number", "123412341234")
            phone = c1.text_input("Phone", "+91 98450 00000")
            emerg = c2.text_input("Emergency contact", "+91 98860 00000")
            itin = st.multiselect("Itinerary", ["MG Road", "Cubbon Park", "Vidhana Soudha", "Lalbagh",
                                                "Bangalore Palace", "Museum Road"], ["MG Road", "Cubbon Park"])
            vf = c1.date_input("Valid from", date(2026, 12, 14))
            vt = c2.date_input("Valid to", date(2026, 12, 21))
            go = st.form_submit_button("Issue digital ID")
        if go:
            try:
                cred = reg.register_tourist(name, nat, doc_type, doc, phone, emerg, itin, vf, vt, "Tourism Department")
                st.session_state.bc_creds.insert(0, cred)
                st.success(f"Issued **{cred.tourist_id}**, sealed in block {len(bc.chain) - 1} "
                           f"(hash `{bc.chain[-1].hash[:20]}…`)")
                q1, q2 = st.columns([1, 2])
                q1.image(credential_qr_png(cred), caption="Credential QR (carried in the tourist app)", width=220)
                q2.json({"credential (tourist keeps)": json.loads(cred.to_json()),
                         "on-chain record (hashes only)": reg.lookup(cred.tourist_id)[1]})
            except ValueError as e:
                st.error(str(e))
    with t2:
        st.write("What a police officer's app does after scanning the QR code.")
        pick = st.selectbox("Scanned credential", creds[:60], index=5,
                            format_func=lambda c: f"{c.tourist_id} · {c.name}"
                            + (" (revoked)" if reg.is_revoked(c.tourist_id) else ""))
        txt = st.text_area("Credential JSON (edit it to simulate a forgery)", pick.to_json(), height=140)
        on = st.date_input("Check on date", date.fromisoformat(pick.valid_from) + timedelta(days=1))
        c1, c2 = st.columns(2)
        if c1.button("Verify"):
            try:
                r = reg.verify(Credential.from_json(txt), on=on)
                (st.success if r.valid else st.error)(f"{'VALID' if r.valid else 'REJECTED'}: {r.reason}"
                                                      f" · record in block {r.block} · incidents on file: {r.incidents}")
                if r.valid and reg.incidents(r.tourist_id):
                    st.dataframe(pd.DataFrame(reg.incidents(r.tourist_id)), hide_index=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"Unreadable credential: {e}")
        if c2.button("Revoke this ID (City Police)"):
            reg.revoke(pick.tourist_id, "revoked from demo", "City Police")
            st.warning(f"{pick.tourist_id} revoked in block {len(bc.chain) - 1}")
    with t3:
        problems = bc.verify_chain()
        (st.success if not problems else st.error)(
            "Chain intact: every hash link, Merkle root and signature checks out." if not problems
            else "Integrity problems: " + "; ".join(problems[:5]))
        st.dataframe(pd.DataFrame([{"block": b.index, "validator": b.validator, "txs": len(b.transactions),
                                    "kinds": ", ".join(sorted({t.kind for t in b.transactions})),
                                    "hash": b.hash[:20] + "…", "prev_hash": b.prev_hash[:20] + "…",
                                    "merkle_root": b.merkle_root[:16] + "…"} for b in bc.chain[::-1]]),
                     hide_index=True, width="stretch", height=360)
        bi = st.number_input("Show block", 0, len(bc.chain) - 1, len(bc.chain) - 1)
        st.json({**bc.chain[bi].header(), "hash": bc.chain[bi].hash,
                 "transactions": [{**t.body(), "tx_hash": t.tx_hash} for t in bc.chain[bi].transactions[:5]]})
    with t4:
        st.write("Change one past incident on a **copy** of the chain and run the integrity check.")
        import copy
        incs = [(b.index, i) for b in bc.chain for i, t in enumerate(b.transactions) if t.kind == "INCIDENT"]
        if not incs:
            st.info("No incidents yet: anchor one from the Geofence simulation page.")
        elif st.button("Rewrite an incident's risk level to LOW"):
            fake = copy.deepcopy(bc)
            bi, ti = incs[0]
            tx = fake.chain[bi].transactions[ti]
            before = tx.payload["risk_level"]
            tx.payload["risk_level"] = "LOW" if before != "LOW" else "CRITICAL"
            st.write(f"Block {bi}: risk level {before} → {tx.payload['risk_level']}")
            for p_ in fake.verify_chain():
                st.error(p_)

# =====================================================================================
elif page == "Results dashboard":
    st.title("Results dashboard")
    ops = pd.read_csv(C.RESULT_DIR / "table4_operating_points.csv")
    sp = ops[ops["Configuration"] == "Hybrid, safety-aware"].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hybrid accuracy", f"{sp['Accuracy (%)']}%", f"{sp['Accuracy (%)'] - ops.iloc[0]['Accuracy (%)']:.1f} pp vs cloud")
    k2.metric("CRITICAL recall", f"{sp['CRITICAL recall (%)']}%")
    k3.metric("Escalated to cloud", f"{sp['Escalated (%)']}%")
    k4.metric("End-to-end latency", f"{sp['Total latency (s)']} s", f"-{sp['Reduction (%)']}%", delta_color="inverse")
    tabs = st.tabs(["Model comparison", "Class imbalance", "ONNX benchmark", "Hybrid trade-off", "Sensitivity", "Stress tests"])
    with tabs[0]:
        st.dataframe(pd.read_csv(C.RESULT_DIR / "table1_models.csv"), hide_index=True, width="stretch")
        st.image(str(C.FIG_DIR / "fig2_model_comparison.png"))
    with tabs[1]:
        st.dataframe(pd.read_csv(C.RESULT_DIR / "table2_imbalance.csv"), hide_index=True, width="stretch")
        st.image(str(C.FIG_DIR / "fig3_imbalance.png"))
        st.image(str(C.FIG_DIR / "fig7_confusion.png"), width=420)
    with tabs[2]:
        st.dataframe(pd.read_csv(C.RESULT_DIR / "table3_benchmark.csv"), hide_index=True, width="stretch")
    with tabs[3]:
        st.dataframe(ops, hide_index=True, width="stretch")
        st.image(str(C.FIG_DIR / "fig4_tradeoff.png"))
        st.image(str(C.FIG_DIR / "fig6_latency_breakdown.png"))
    with tabs[4]:
        st.image(str(C.FIG_DIR / "fig5_sensitivity.png"))
    with tabs[5]:
        st.dataframe(pd.read_csv(C.RESULT_DIR / "table5_stress.csv"), hide_index=True, width="stretch")

# =====================================================================================
elif page == "System validation & tests":
    st.title("System validation & testing")
    vp = C.RESULT_DIR / "system_validation.json"
    if vp.exists():
        V = json.load(open(vp))
        st.subheader("Architecture validation")
        st.json(V["architecture"])
        st.subheader("Scenario execution")
        st.dataframe(pd.DataFrame(V["scenarios"]), hide_index=True, width="stretch")
        st.write("Offline mode:", V["offline_mode"])
        st.subheader("Integrated system vs simulation (4,000 test events)")
        b = V["bulk"]
        c1, c2, c3 = st.columns(3)
        c1.metric("System accuracy", f"{b['system_accuracy']*100:.1f}%", f"sim {b['simulation_accuracy']*100:.1f}%", delta_color="off")
        c2.metric("CRITICAL recall", f"{b['system_critical_recall']*100:.1f}%", f"sim {b['simulation_critical_recall']*100:.1f}%", delta_color="off")
        c3.metric("On-device latency (mean)", f"{b['on_device_latency_mean_us']:.0f} µs")
        if "blockchain" in V:
            st.subheader("Blockchain digital ID validation")
            st.json(V["blockchain"])
    else:
        st.warning("Run `python validate_system.py` to produce the validation report.")
    tp = C.RESULT_DIR / "test_summary.json"
    if tp.exists():
        st.subheader("Automated test suite (pytest)")
        T = json.load(open(tp))
        st.metric("Tests passed", f"{T['passed']} / {T['total']}")
        st.dataframe(pd.DataFrame(T["by_category"]), hide_index=True, width="stretch")
    else:
        st.info("Run `python run_tests.py` to produce the test report.")

# =====================================================================================
else:
    st.title("Smart Tourist Safety Monitoring & Incident Response System")
    st.caption("Using AI, Geo-Fencing and Blockchain-Based Digital ID")
    st.image(str(C.FIG_DIR / "fig1_architecture.png"), width=620)
    st.markdown("""
**Problem.** In STSMIRS, the cloud LLM call takes 6.8 s of a 12.7 s alert response time.

**Idea.** Classify each Stage-1 event on the phone with a tiny model; send only uncertain or
severe events to the cloud LLM (selective escalation).

**Digital ID.** Tourists get a blockchain-anchored digital ID at registration; police verify it by
scanning a QR code, and every AI alert is recorded on the chain as a tamper-evident audit trail.

**Pipeline.** GPS fixes → geofence gate (Haversine, zones, route deviation, idle ≥ 15 min)
→ on-device ONNX logistic regression (0.7 KB) → escalation policy (θc + severity)
→ cloud LLM verification → alert and action.

**Important.** All data are synthetic and the cloud LLM is a calibrated stand-in, so the numbers
show relative trade-offs, not field performance.
""")
