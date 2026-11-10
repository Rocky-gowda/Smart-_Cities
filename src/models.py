"""On-device candidate models, their hyper-parameter grids and simple baselines."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED, CLASS_MIX


class RuleScore(BaseEstimator, ClassifierMixin):
    """Hand-crafted rule engine: log displacement + hazard flag + night indicator,
    cut at training-set class quantiles (an 'extended Stage-1' baseline)."""

    def _score(self, X):
        log_disp, hour_sin, hour_cos, hazard = X[:, 0], X[:, 2], X[:, 3], X[:, 4]
        hour = (np.degrees(np.arctan2(hour_sin, hour_cos)) / 15.0) % 24
        night = ((hour >= 22) | (hour < 5)).astype(float)
        return log_disp + hazard + night

    def fit(self, X, y):
        self.classes_ = np.arange(4)
        mix = np.bincount(y, minlength=4) / len(y)
        self.cuts_ = np.quantile(self._score(X), np.cumsum(mix)[:-1])
        return self

    def predict(self, X):
        return np.digitize(self._score(X), self.cuts_)


def scaled(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def candidates():
    """name -> (estimator, param_grid). Grids are kept small enough for a laptop."""
    return {
        "Majority class": (DummyClassifier(strategy="most_frequent"), {}),
        "Rule-based score": (RuleScore(), {}),
        "Logistic regression": (scaled(LogisticRegression(max_iter=3000)),
                                {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "Decision tree": (DecisionTreeClassifier(random_state=SEED),
                          {"max_depth": [4, 6, 8, 10], "min_samples_leaf": [5, 20, 50]}),
        "Random forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                          {"n_estimators": [200], "max_depth": [8, 12, None],
                           "min_samples_leaf": [5, 20]}),
        "Gradient boosting": (GradientBoostingClassifier(random_state=SEED),
                              {"n_estimators": [100, 200], "max_depth": [2, 3],
                               "learning_rate": [0.05, 0.1]}),
        "Hist. gradient boosting": (HistGradientBoostingClassifier(random_state=SEED),
                                    {"max_iter": [100, 200], "learning_rate": [0.05, 0.1],
                                     "max_depth": [3, None]}),
        "MLP (32-16)": (scaled(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600,
                                             early_stopping=True, random_state=SEED)),
                        {"clf__alpha": [1e-4, 1e-3, 1e-2]}),
    }


LEARNED = ["Logistic regression", "Decision tree", "Random forest", "Gradient boosting",
           "Hist. gradient boosting", "MLP (32-16)"]


def balanced_lr(C):
    return scaled(LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))


def work_per_inference(name, model, X):
    """Hardware-independent computation proxy: MACs for linear/MLP models,
    mean tree-node comparisons for tree models."""
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if name == "Logistic regression":
        return f"{est.coef_.size} MAC"
    if name.startswith("MLP"):
        sizes = [X.shape[1], *est.hidden_layer_sizes, len(est.classes_)]
        return f"{sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))} MAC"
    if name == "Decision tree":
        return f"{est.decision_path(X).sum(axis=1).mean() - 1:.0f}"
    if name == "Random forest":
        return f"{sum(t.decision_path(X).sum(axis=1).mean() - 1 for t in est.estimators_):,.0f}"
    if name == "Gradient boosting":
        return f"{sum(t.decision_path(X).sum(axis=1).mean() - 1 for t in est.estimators_.ravel()):,.0f}"
    return "–"
