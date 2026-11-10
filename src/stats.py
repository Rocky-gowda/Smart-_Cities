"""Evaluation metrics and statistical tests."""
import numpy as np
from scipy import stats as st
from sklearn.metrics import accuracy_score, f1_score, recall_score

N_CLASSES = 4


def metrics(y, p):
    rec = recall_score(y, p, labels=range(N_CLASSES), average=None, zero_division=0)
    return dict(accuracy=accuracy_score(y, p),
                macro_f1=f1_score(y, p, labels=range(N_CLASSES), average="macro", zero_division=0),
                recall=rec.tolist(), critical_recall=float(rec[3]))


def stratified_bootstrap_idx(y, n_boot, rng):
    """Yield index arrays that resample within each class."""
    groups = [np.where(y == c)[0] for c in range(N_CLASSES)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])


def bootstrap_ci(y, p, n_boot=2000, seed=0):
    """95% percentile CIs for accuracy and CRITICAL recall."""
    rng = np.random.default_rng(seed)
    acc, crit, f1 = [], [], []
    for idx in stratified_bootstrap_idx(y, n_boot, rng):
        yy, pp = y[idx], p[idx]
        acc.append((yy == pp).mean())
        m = yy == 3
        crit.append((pp[m] == 3).mean())
        f1.append(f1_score(yy, pp, average="macro", labels=range(N_CLASSES), zero_division=0))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(accuracy_ci=q(acc), critical_recall_ci=q(crit), macro_f1_ci=q(f1))


def mcnemar_exact(y, p1, p2):
    """Exact (binomial) McNemar test on paired predictions."""
    a = (p1 == y) & (p2 != y)
    b = (p1 != y) & (p2 == y)
    n01, n10 = int(a.sum()), int(b.sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(st.binomtest(min(n01, n10), n, 0.5).pvalue)


def five_by_two_cv_f(make_a, make_b, X, y, seed=0):
    """Alpaydin's combined 5x2 cv F-test. make_* return fresh unfitted models."""
    from sklearn.model_selection import StratifiedKFold
    diffs = np.zeros((5, 2))
    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + i)
        for j, (tr, te) in enumerate(skf.split(X, y)):
            ea = 1 - (make_a().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()
            eb = 1 - (make_b().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()
            diffs[i, j] = ea - eb
    mean = diffs.mean(axis=1, keepdims=True)
    s2 = ((diffs - mean) ** 2).sum(axis=1)
    f = (diffs ** 2).sum() / (2 * s2.sum())
    return float(1 - st.f.cdf(f, 10, 5))


def expected_calibration_error(proba, y, n_bins=15):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def min_cost_predict(proba, under_weights):
    """Minimum-expected-cost decision with the asymmetric cost matrix."""
    k = proba.shape[1]
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            C[i, j] = (i - j) * under_weights[i] if i > j else (j - i)
    return (proba @ C).argmin(axis=1)
