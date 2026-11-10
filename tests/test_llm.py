"""Unit tests for the cloud LLM connector: request format per provider, reply parsing, fallback."""
import json

import httpx
import pytest

from server.llm import CloudVerifier, build_prompt, parse_reply, rules_decision

pytestmark = pytest.mark.unit
EVENT = {"t": "2026-11-01T01:00", "hour": 1.0, "idle_min": 40, "displacement_m": 900, "zones": ["Lake"],
         "hazard_flag": 1, "local_class": "HIGH", "local_confidence": 0.6}
CTX = {"nearby_incidents": 0, "tourist_incidents": 0, "nationality": "IN"}


def capture(reply_json):
    seen = {}

    def handler(req):
        seen["url"], seen["headers"], seen["body"] = str(req.url), dict(req.headers), json.loads(req.content)
        return httpx.Response(200, json=reply_json)
    return seen, httpx.MockTransport(handler)


def test_anthropic_request_and_reply():
    seen, tr = capture({"content": [{"type": "text", "text": '{"risk_level": "CRITICAL", "reason": "night, hazard"}'}]})
    d = CloudVerifier("anthropic", "claude-x", "KEY", transport=tr).verify(EVENT, CTX)
    assert d.risk_level == "CRITICAL" and not d.fallback
    assert seen["url"] == "https://api.anthropic.com/v1/messages" and seen["headers"]["x-api-key"] == "KEY"
    assert seen["body"]["model"] == "claude-x"


def test_openai_request_and_reply():
    seen, tr = capture({"choices": [{"message": {"content": '```json\n{"risk_level":"HIGH","reason":"r"}\n```'}}]})
    d = CloudVerifier("openai", "gpt-x", "KEY", transport=tr).verify(EVENT, CTX)
    assert d.risk_level == "HIGH" and seen["headers"]["authorization"] == "Bearer KEY"


def test_gemini_request_and_reply():
    seen, tr = capture({"candidates": [{"content": {"parts": [{"text": '{"risk_level":"MODERATE","reason":"r"}'}]}}]})
    d = CloudVerifier("gemini", "gem-x", "KEY", transport=tr).verify(EVENT, CTX)
    assert d.risk_level == "MODERATE" and "gem-x:generateContent" in seen["url"] and "key=KEY" in seen["url"]


def test_http_error_falls_back():
    tr = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad key"}))
    d = CloudVerifier("anthropic", "m", "bad", transport=tr).verify(EVENT, CTX)
    assert d.fallback and d.provider == "rules engine (fallback)" and "HTTPStatusError" in d.reason


def test_unreadable_reply_falls_back():
    tr = httpx.MockTransport(lambda req: httpx.Response(200, json={"content": [{"text": "I cannot help"}]}))
    assert CloudVerifier("anthropic", "m", "k", transport=tr).verify(EVENT, CTX).fallback


@pytest.mark.parametrize("text,level", [('{"risk_level":"low","reason":"x"}', "LOW"),
                                        ("Risk: HIGH because it is night", "HIGH"),
                                        ('{"risk_level":"EXTREME"}', None)])
def test_parse_reply(text, level):
    r = parse_reply(text)
    assert (r["risk_level"] if r else None) == level


def test_prompt_contains_context():
    p = build_prompt(EVENT, {**CTX, "nearby_incidents": 3})
    assert "40 minutes" in p and "900 m" in p and "Lake" in p and "last 24 h: 3" in p


def test_rules_engine():
    assert rules_decision(EVENT, CTX)["risk_level"] == "CRITICAL"            # hazard at night raises HIGH
    day = {**EVENT, "hour": 13, "hazard_flag": 0, "displacement_m": 50, "local_class": "HIGH"}
    assert rules_decision(day, CTX)["risk_level"] == "MODERATE"             # daytime, on route, no hazard
