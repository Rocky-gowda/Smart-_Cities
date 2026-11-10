"""Server configuration, read from environment variables (all optional).

  DATA_DIR            where the database, blockchain and keys are stored (default: server_data/)
  ADMIN_USER          police/admin login name (default: admin)
  ADMIN_PASSWORD      police/admin password (default: admin123  <- change it!)
  SECRET_KEY          signing key for login tokens (auto-generated and saved if not set)
  IDLE_MINUTES        Stage-1 idle threshold (default 15)
  TIME_SCALE          demo time acceleration: 1 real minute counts as TIME_SCALE minutes (default 1)
  ESCALATION_POLICY   safety | conf   (default safety)
  THETA               confidence threshold (default: the value tuned in results/results.json)
  LLM_PROVIDER        anthropic | openai | gemini | rules | auto (default auto: first key found)
  ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY    cloud LLM credentials
  LLM_MODEL           model name for the chosen provider (sensible default per provider)
  LLM_TIMEOUT         seconds before the cloud call is abandoned (default 15)
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini", "gemini": "gemini-2.0-flash"}


def _tuned_theta() -> float:
    try:
        res = json.loads((ROOT / "results" / "results.json").read_text())
        return next(o for o in res["operating_points"] if o["config"] == "Hybrid, safety-aware")["theta"]
    except Exception:  # noqa: BLE001
        return 0.725


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT / "server_data")))
    admin_user: str = field(default_factory=lambda: os.getenv("ADMIN_USER", "admin"))
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "admin123"))
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", ""))
    idle_minutes: float = field(default_factory=lambda: float(os.getenv("IDLE_MINUTES", "15")))
    time_scale: float = field(default_factory=lambda: float(os.getenv("TIME_SCALE", "1")))
    policy: str = field(default_factory=lambda: os.getenv("ESCALATION_POLICY", "safety"))
    theta: float = field(default_factory=lambda: float(os.getenv("THETA", _tuned_theta())))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "15")))
    api_keys: dict = field(default_factory=lambda: {
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""), "openai": os.getenv("OPENAI_API_KEY", ""),
        "gemini": os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))})
    model_json: Path = ROOT / "models" / "deployed_model.json"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.secret_key:
            p = self.data_dir / "secret.key"
            if not p.exists():
                p.write_text(secrets.token_hex(32))
            self.secret_key = p.read_text().strip()
        if self.llm_provider == "auto":
            self.llm_provider = next((k for k in ("anthropic", "openai", "gemini") if self.api_keys.get(k)), "rules")
        if not self.llm_model and self.llm_provider in DEFAULT_MODELS:
            self.llm_model = DEFAULT_MODELS[self.llm_provider]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"
