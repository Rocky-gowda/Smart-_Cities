"""Hybrid edge-cloud escalation: cloud models, policies and the latency model."""
import numpy as np

from .config import CLOUD_LATENCY_MEAN, CLOUD_LATENCY_CV, FIXED_STAGES, CLOUD_TARGET_ACC


# ------------------------------------------------------------------ cloud models
def cloud_proxy_predict(latent_det, cuts, sigma, rng):
    """Case-dependent proxy: the cloud sees the full-information latent score
    (including context the phone cannot observe) plus Gaussian reasoning noise.
    Its errors therefore fall near class boundaries."""
    return np.digitize(latent_det + rng.normal(0, sigma, len(latent_det)), cuts)


def calibrate_sigma(latent_det, y, cuts, target=CLOUD_TARGET_ACC, seed=0, reps=20):
    """Bisection on sigma so the proxy's mean accuracy equals the reported 89.6%."""
    def acc(s):
        rng = np.random.default_rng(seed)
        return np.mean([(cloud_proxy_predict(latent_det, cuts, s, rng) == y).mean()
                        for _ in range(reps)])
    lo, hi = 0.0, 2.0
    if acc(lo) < target:
        raise ValueError("Cloud cannot reach target accuracy even without noise.")
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if acc(mid) > target else (lo, mid)
    return (lo + hi) / 2


def cloud_independent_predict(y, p_correct, rng):
    """Each case correct with fixed probability; otherwise a random wrong class."""
    correct = rng.random(len(y)) < p_correct
    wrong = (y + rng.integers(1, 4, len(y))) % 4
    return np.where(correct, y, wrong)


# ------------------------------------------------------------------ policies
def escalate_mask(proba, theta, policy, pred=None):
    """confidence-only: escalate if max prob < theta.
    safety-aware: additionally escalate every local HIGH/CRITICAL prediction.
    `pred` overrides the local prediction (used by the cost-sensitive rule)."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1) if pred is None else pred
    esc = conf < theta
    if policy == "safety":
        esc = esc | (pred >= 2)
    return esc


def combine(local_pred, cloud_pred, esc):
    return np.where(esc, cloud_pred, local_pred)


# ------------------------------------------------------------------ latency
def cloud_latency_draws(n, rng, mean=CLOUD_LATENCY_MEAN, cv=CLOUD_LATENCY_CV):
    s2 = np.log(1 + cv ** 2)
    return rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), n)


def ai_stage_latency(esc, local_s, rng):
    """Mean AI-stage latency (s): every event runs locally, escalated ones also wait for the cloud."""
    return float(local_s + (esc * cloud_latency_draws(len(esc), rng)).mean())


def total_latency(ai_s):
    return ai_s + FIXED_STAGES


# ------------------------------------------------------------------ evaluation helpers
def evaluate_point(proba, y, latent_det, cuts, sigma, theta, policy, local_s,
                   n_mc, seed, cloud="proxy", local_pred=None):
    """Average accuracy / CRITICAL recall / latency over Monte Carlo cloud draws."""
    rng = np.random.default_rng(seed)
    local = proba.argmax(axis=1) if local_pred is None else local_pred
    esc = escalate_mask(proba, theta, policy, local)
    accs, crits, lats, cloud_acc_esc = [], [], [], []
    for _ in range(n_mc):
        if cloud == "proxy":
            cp = cloud_proxy_predict(latent_det, cuts, sigma, rng)
        else:
            cp = cloud_independent_predict(y, CLOUD_TARGET_ACC, rng)
        final = combine(local, cp, esc)
        accs.append((final == y).mean())
        crits.append((final[y == 3] == 3).mean())
        lats.append(ai_stage_latency(esc, local_s, rng))
        if esc.any():
            cloud_acc_esc.append((cp[esc] == y[esc]).mean())
    ai = float(np.mean(lats))
    return dict(theta=theta, policy=policy, escalated=float(esc.mean()),
                accuracy=float(np.mean(accs)), critical_recall=float(np.mean(crits)),
                ai_latency_s=ai, total_latency_s=total_latency(ai),
                cloud_acc_on_escalated=float(np.mean(cloud_acc_esc)) if cloud_acc_esc else None)


def ci_point(proba, y, latent_det, cuts, sigma, theta, policy, n_draws, seed):
    """CI combining test-set bootstrap and Monte Carlo cloud variation."""
    rng = np.random.default_rng(seed)
    local = proba.argmax(axis=1)
    esc = escalate_mask(proba, theta, policy)
    groups = [np.where(y == c)[0] for c in range(4)]
    accs, crits = [], []
    for _ in range(n_draws):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        cp = cloud_proxy_predict(latent_det[idx], cuts, sigma, rng)
        final = combine(local[idx], cp, esc[idx])
        yy = y[idx]
        accs.append((final == yy).mean())
        crits.append((final[yy == 3] == 3).mean())
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(accuracy_ci=q(accs), critical_recall_ci=q(crits))


def sensitivity_sweep(proba, y, theta, policy, accs_on_esc, n_mc=200, seed=0):
    """System accuracy / CRITICAL recall when the cloud's accuracy on escalated
    cases is set directly (independent errors)."""
    rng = np.random.default_rng(seed)
    local = proba.argmax(axis=1)
    esc = escalate_mask(proba, theta, policy)
    out = []
    for q in accs_on_esc:
        a, c = [], []
        for _ in range(n_mc):
            cp = cloud_independent_predict(y, q, rng)
            final = combine(local, cp, esc)
            a.append((final == y).mean())
            c.append((final[y == 3] == 3).mean())
        out.append(dict(cloud_acc=q, accuracy=float(np.mean(a)), critical_recall=float(np.mean(c))))
    return out
