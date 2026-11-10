"""The integrated hybrid edge-cloud risk system (architecture execution).

Pipeline for every GPS stream:
    GPS fixes -> Stage 1 geofence gate -> Stage 2a on-device ONNX model
              -> escalation policy -> [Stage 2b cloud LLM] -> alert record

The cloud LLM of STSMIRS cannot be called from here, so `CloudLLMStub` stands in
for it: it uses the case-dependent proxy from the simulation study (it sees the
full context, plus calibrated noise) and draws a latency from the reported
distribution. Swap it for a real API client to deploy.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .generator import latent_score
from .geofence import CandidateEvent, GeofenceGate
from .hybrid import cloud_latency_draws

SEVERITY_ACTION = {
    "LOW": "Log only",
    "MODERATE": "Push safety check-in to tourist",
    "HIGH": "Notify nearest tourist-police unit",
    "CRITICAL": "Dispatch responders + alert emergency contact",
}


class OnDeviceModel:
    """Stage 2a: the exported, class-balanced logistic-regression model (ONNX)."""

    def __init__(self, path=C.MODEL_DIR / "deployed_model.onnx"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.sess.run(None, {self.input: x.astype(np.float32)})[1])


class CloudLLMStub:
    """Stage 2b stand-in for the cloud LLM (see module docstring)."""

    def __init__(self, sigma: float, cuts, seed: int = 0, available: bool = True):
        self.sigma, self.cuts = sigma, np.asarray(cuts)
        self.rng = np.random.default_rng(seed)
        self.available = available

    def classify(self, event: CandidateEvent, context: Dict) -> tuple[int, float]:
        if not self.available:
            raise ConnectionError("cloud unreachable")
        s = latent_score(np.array([max(context.get("disp_true_m", event.displacement_m), 1.0)]), np.array([event.idle_min]),
                         np.array([event.hour]), np.array([context.get("hazard_true", event.hazard_flag)]),
                         np.array([context.get("context_z", 0.0)]), np.array([context.get("incident", 0)]))
        label = int(np.digitize(s + self.rng.normal(0, self.sigma), self.cuts)[0])
        return label, float(cloud_latency_draws(1, self.rng)[0])


@dataclass
class Alert:
    tourist_id: str
    time: str
    lat: float
    lon: float
    zones: List[str]
    features: Dict[str, float]
    local_class: str
    local_confidence: float
    escalated: bool
    escalation_reason: str
    final_class: str
    decided_by: str
    action: str
    timings_s: Dict[str, float] = field(default_factory=dict)
    digital_id: Optional[Dict] = None          # blockchain ID check (if a registry is attached)
    chain_record: Optional[str] = None         # hash of the INCIDENT transaction on the chain


class HybridRiskSystem:
    def __init__(self, theta: Optional[float] = None, policy: str = "safety",
                 cloud: Optional[CloudLLMStub] = None, model: Optional[OnDeviceModel] = None,
                 registry=None, chain_issuer: str = "City Police"):
        self.registry, self.chain_issuer = registry, chain_issuer
        res = json.load(open(C.RESULT_DIR / "results.json"))
        cuts = json.load(open(C.DATA_DIR / "cuts.json"))["cuts"]
        sp = next(o for o in res["operating_points"] if o["config"] == "Hybrid, safety-aware")
        self.theta = sp["theta"] if theta is None else theta
        self.policy = policy
        self.model = model or OnDeviceModel()
        self.cloud = cloud or CloudLLMStub(res["cloud_sigma"], cuts)
        self.offline_queue: deque = deque()

    def decide(self, event: CandidateEvent, context: Optional[Dict] = None) -> Alert:
        context = context or {}
        timings = {}
        x = event.features()
        t0 = time.perf_counter()
        proba = self.model.predict_proba(x)[0]
        timings["stage2a_on_device"] = time.perf_counter() - t0
        local = int(proba.argmax())
        conf = float(proba.max())

        reasons = []
        if conf < self.theta:
            reasons.append(f"confidence {conf:.2f} < theta {self.theta:.3f}")
        if self.policy == "safety" and local >= 2:
            reasons.append(f"local prediction {C.CLASSES[local]} needs verification")
        escalate = bool(reasons)

        final, by = local, "on-device"
        timings["stage2b_cloud"] = 0.0
        if escalate:
            try:
                final, lat = self.cloud.classify(event, context)
                timings["stage2b_cloud"] = lat
                by = "cloud LLM"
            except ConnectionError:
                self.offline_queue.append(event)
                by = "on-device (cloud offline, escalation queued)"
        timings["ai_stage_total"] = timings["stage2a_on_device"] + timings["stage2b_cloud"]
        timings["end_to_end_estimate"] = timings["ai_stage_total"] + C.FIXED_STAGES

        names = ["log_disp", "log_idle", "hour_sin", "hour_cos", "hazard_flag"]
        return Alert(
            tourist_id=event.tourist_id, time=event.t.isoformat(), lat=event.lat, lon=event.lon,
            zones=event.zones, features=dict(zip(names, map(float, x[0]))) | {
                "displacement_m": event.displacement_m, "idle_min": event.idle_min, "hour": event.hour},
            local_class=C.CLASSES[local], local_confidence=conf, escalated=escalate,
            escalation_reason="; ".join(reasons) or "confident, low severity",
            final_class=C.CLASSES[final], decided_by=by, action=SEVERITY_ACTION[C.CLASSES[final]],
            timings_s=timings)

    def run_stream(self, tourist_id, fixes, route, zones, context=None, credential=None) -> List[Alert]:
        """Run one tourist's GPS stream end to end. With a blockchain registry attached,
        the tourist's digital ID is verified first and every alert is anchored on the chain."""
        id_check = None
        if self.registry is not None and credential is not None:
            t0 = time.perf_counter()
            v = self.registry.verify(credential, on=fixes[0].t.date() if fixes else None)
            id_check = {"tourist_id": v.tourist_id, "valid": v.valid, "reason": v.reason, "block": v.block,
                        "verify_s": time.perf_counter() - t0}
            tourist_id = credential.tourist_id
        gate = GeofenceGate(tourist_id, route, zones)
        alerts = []
        for f in fixes:
            t0 = time.perf_counter()
            ev = gate.update(f)
            dt = time.perf_counter() - t0
            if ev is not None:
                a = self.decide(ev, context)
                a.timings_s["stage1_gate"] = dt
                a.digital_id = id_check
                if self.registry is not None:
                    t1 = time.perf_counter()
                    tx = self.registry.record_incident(tourist_id, asdict(a), self.chain_issuer)
                    a.timings_s["blockchain_record"] = time.perf_counter() - t1
                    a.chain_record = tx.tx_hash
                alerts.append(a)
        if self.registry is not None and alerts:
            self.registry.chain.seal_block()
        return alerts


def alert_to_dict(a: Alert) -> Dict:
    return asdict(a)
