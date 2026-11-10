"""Stage 2b: cloud verification of escalated events.

Providers
  anthropic / openai / gemini : a real LLM is asked to assign the risk level, using the event
                                plus server-side context the phone does not have (recent incidents
                                nearby, the tourist's history, zone names).
  rules                         : used when no API key is configured, or as the fallback when the
                                LLM call fails or times out. A transparent context rule engine.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx

CLASSES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

SYSTEM = ("You are the risk-assessment engine of a tourist-safety monitoring system used by police. "
          "Assign exactly one risk level to a safety event: LOW, MODERATE, HIGH or CRITICAL. "
          "LOW = routine stop, no concern. MODERATE = unusual, send the tourist a check-in. "
          "HIGH = likely in difficulty, notify the nearest police unit. "
          "CRITICAL = possible emergency, dispatch responders. "
          'Reply with JSON only: {"risk_level": "...", "reason": "<one sentence>"}')


@dataclass
class CloudDecision:
    risk_level: str
    reason: str
    provider: str
    latency_s: float
    fallback: bool = False


def build_prompt(event: Dict, context: Dict) -> str:
    return (
        "Safety event detected by the geofence on a tourist's phone:\n"
        f"- local time: {event.get('t')} (hour {event.get('hour', 0):.1f})\n"
        f"- idle (not moving) for {event.get('idle_min', 0):.0f} minutes\n"
        f"- {event.get('displacement_m', 0):.0f} m away from the planned itinerary route\n"
        f"- inside geofenced zones: {', '.join(event.get('zones') or []) or 'none'}"
        f" (hazard zone: {'yes' if event.get('hazard_flag') else 'no'})\n"
        f"- on-device model: {event.get('local_class')} with confidence {event.get('local_confidence', 0):.2f}\n"
        "Context known to the server:\n"
        f"- incidents reported within 500 m in the last 24 h: {context.get('nearby_incidents', 0)}\n"
        f"- this tourist's previous incidents: {context.get('tourist_incidents', 0)}\n"
        f"- tourist nationality: {context.get('nationality', 'unknown')}\n"
        "Assign the risk level.")


def parse_reply(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            lvl = str(d.get("risk_level", "")).upper().strip()
            if lvl in CLASSES:
                return {"risk_level": lvl, "reason": str(d.get("reason", ""))[:300]}
        except json.JSONDecodeError:
            pass
    for lvl in reversed(CLASSES):            # last resort: a bare level word in the text
        if re.search(rf"\b{lvl}\b", text.upper()):
            return {"risk_level": lvl, "reason": text.strip()[:300]}
    return None


def rules_decision(event: Dict, context: Dict) -> Dict:
    """Context rules used without an LLM: start from the on-device class and raise it when
    server-side context points to more danger."""
    level = CLASSES.index(event.get("local_class", "MODERATE"))
    reasons = [f"on-device {event.get('local_class')} ({event.get('local_confidence', 0):.2f})"]
    hour = event.get("hour", 12)
    night = hour >= 22 or hour < 5
    if context.get("nearby_incidents", 0) >= 2:
        level += 1
        reasons.append(f"{context['nearby_incidents']} recent incidents nearby")
    if event.get("hazard_flag") and night and level < 3:
        level += 1
        reasons.append("hazard zone at night")
    if event.get("idle_min", 0) >= 60 and event.get("displacement_m", 0) > 1000 and level < 3:
        level += 1
        reasons.append("long stop far from route")
    if not event.get("hazard_flag") and not night and event.get("displacement_m", 0) < 150 and \
            context.get("nearby_incidents", 0) == 0:
        level = min(level, 1)
        reasons.append("daytime stop on route, no hazards")
    return {"risk_level": CLASSES[min(level, 3)], "reason": "; ".join(reasons)}


class CloudVerifier:
    def __init__(self, provider: str, model: str = "", api_key: str = "", timeout: float = 15.0,
                 transport: Optional[httpx.BaseTransport] = None):
        self.provider, self.model, self.api_key, self.timeout = provider, model, api_key, timeout
        self.client = httpx.Client(timeout=timeout, transport=transport)

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}" if self.provider != "rules" else "rules engine"

    def _call(self, prompt: str) -> str:
        if self.provider == "anthropic":
            r = self.client.post("https://api.anthropic.com/v1/messages",
                                 headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"},
                                 json={"model": self.model, "max_tokens": 200, "system": SYSTEM,
                                       "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", []))
        if self.provider == "openai":
            r = self.client.post("https://api.openai.com/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {self.api_key}"},
                                 json={"model": self.model, "max_tokens": 200, "temperature": 0,
                                       "messages": [{"role": "system", "content": SYSTEM},
                                                    {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if self.provider == "gemini":
            r = self.client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
                      "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0, "maxOutputTokens": 200}})
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise ValueError(f"unknown provider {self.provider}")

    def verify(self, event: Dict, context: Dict) -> CloudDecision:
        t0 = time.perf_counter()
        if self.provider != "rules":
            try:
                parsed = parse_reply(self._call(build_prompt(event, context)))
                if parsed:
                    return CloudDecision(parsed["risk_level"], parsed["reason"], self.name, time.perf_counter() - t0)
                err = "unreadable LLM reply"
            except Exception as e:  # noqa: BLE001  (network error, timeout, bad key, quota...)
                err = f"{type(e).__name__}"
            d = rules_decision(event, context)
            return CloudDecision(d["risk_level"], f"LLM unavailable ({err}); rules: {d['reason']}",
                                 "rules engine (fallback)", time.perf_counter() - t0, fallback=True)
        d = rules_decision(event, context)
        return CloudDecision(d["risk_level"], d["reason"], "rules engine", time.perf_counter() - t0)
