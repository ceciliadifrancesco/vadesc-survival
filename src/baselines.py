"""Standard survival baselines."""
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .metrics import survival_metrics


def _sksurv_surv_fn(model, X):
    return lambda ts: np.vstack([f(ts) for f in model.predict_survival_function(X)])


def run_cox(d, alpha=0.1):
    m = CoxPHSurvivalAnalysis(alpha=alpha, ties="efron").fit(d.X_tr, d.y_tr)
    return survival_metrics(d, m.predict(d.X_te), _sksurv_surv_fn(m, d.X_te))


def run_rsf(d, seed=42):
    m = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, max_features="sqrt",
                             n_jobs=-1, random_state=seed).fit(d.X_tr, d.y_tr)
    return survival_metrics(d, m.predict(d.X_te), _sksurv_surv_fn(m, d.X_te))


def run_weibull_aft(d, penalizer=0.05):
    keep = d.X_tr.std(0) > 0
    cols = [f"x{i}" for i in range(keep.sum())]
    df = pd.DataFrame(d.X_tr[:, keep], columns=cols)
    df["T"], df["E"] = np.maximum(d.t_tr, 1e-3), d.e_tr
    m = WeibullAFTFitter(penalizer=penalizer).fit(df, "T", "E")
    X_te = pd.DataFrame(d.X_te[:, keep], columns=cols)
    surv_fn = lambda ts: m.predict_survival_function(X_te, times=ts).to_numpy().T
    return survival_metrics(d, -m.predict_median(X_te).to_numpy().ravel(), surv_fn)
