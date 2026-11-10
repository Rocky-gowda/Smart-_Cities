"""Synthetic post-gate event generator.

Each row is one event that has already passed the Stage-1 geofence / idle gate.
The generator creates
  * ground-truth variables (true displacement, idle time, hour, hazard zone),
  * two variables only the cloud LLM can see (context score z, nearby incident),
  * a latent risk score, cut at quantiles that reproduce the published class mix,
  * the degraded inputs the phone actually observes.

It is a controlled test bed, not a substitute for real field data.
"""
import numpy as np
import pandas as pd

from .config import GEN, CLASS_MIX, FEATURES


def night_profile(hour):
    """Smooth 0..1 profile peaking at 02:00 and lowest at 14:00."""
    return 0.5 * (1 + np.cos(2 * np.pi * (hour - 2.0) / 24.0))


def latent_score(d, idle, hour, hazard, z, incident, g=GEN):
    """Deterministic part of the risk score (everything except noise)."""
    return (g["b_disp"] * np.log(d)
            + g["b_night"] * night_profile(hour)
            + g["b_hazard"] * hazard
            + g["b_idle"] * np.log(idle)
            + g["b_context"] * z
            + g["b_incident"] * incident)


def class_cuts(score, mix=CLASS_MIX):
    """Score thresholds that reproduce the target class proportions."""
    return np.quantile(score, np.cumsum(mix)[:-1])


def generate(n, seed, cuts=None, gps_log_sd=None, hazard_flag_error=None, g=GEN):
    """Generate n events. If `cuts` is None they are fitted on this sample.

    Returns (DataFrame, cuts). Stress tests pass custom noise levels and the
    cuts of the original training distribution so labels mean the same thing.
    """
    rng = np.random.default_rng(seed)
    gps_log_sd = g["gps_log_sd"] if gps_log_sd is None else gps_log_sd
    hazard_flag_error = g["hazard_flag_error"] if hazard_flag_error is None else hazard_flag_error

    d = np.exp(rng.normal(g["disp_log_mean"], g["disp_log_sd"], n))             # metres
    idle = g["idle_min"] + rng.gamma(g["idle_gamma_k"], g["idle_gamma_theta"], n)  # minutes
    hour = rng.uniform(0, 24, n)
    hazard = (rng.random(n) < g["hazard_prev"]).astype(int)
    z = rng.normal(0, 1, n)
    incident = (rng.random(n) < g["incident_prev"]).astype(int)

    s_det = latent_score(d, idle, hour, hazard, z, incident, g)
    s = s_det + rng.normal(0, g["noise_sd"], n)
    if cuts is None:
        cuts = class_cuts(s)
    label = np.digitize(s, cuts)

    # what the device observes
    d_obs = np.maximum(d * np.exp(rng.normal(0, gps_log_sd, n)), g["gps_floor_m"])
    flip = rng.random(n) < hazard_flag_error
    hazard_flag = np.where(flip, 1 - hazard, hazard)

    df = pd.DataFrame({
        "disp_true_m": d, "idle_min": idle, "hour": hour, "hazard_true": hazard,
        "context_z": z, "incident": incident,
        "latent_det": s_det, "latent": s, "label": label,
        "disp_obs_m": d_obs, "hazard_flag": hazard_flag,
    })
    df = add_features(df)
    return df, cuts


def add_features(df):
    """Device-side features (the only inputs the on-device model gets)."""
    df["log_disp"] = np.log(df["disp_obs_m"])
    df["log_idle"] = np.log(df["idle_min"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def features_from_raw(disp_m, idle_min, hour, hazard_flag):
    """Build one feature row from raw inputs (used by the demo app)."""
    return np.array([[np.log(max(disp_m, GEN["gps_floor_m"])), np.log(idle_min),
                      np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                      float(hazard_flag)]], dtype=np.float32)


def split(df, seed, n_train, n_val, n_test):
    """Stratified train / validation / test split."""
    from sklearn.model_selection import train_test_split
    tr, rest = train_test_split(df, train_size=n_train, stratify=df["label"], random_state=seed)
    va, te = train_test_split(rest, train_size=n_val, test_size=n_test,
                              stratify=rest["label"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


__all__ = ["generate", "split", "features_from_raw", "latent_score", "class_cuts", "FEATURES"]
