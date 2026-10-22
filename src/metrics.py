"""Evaluation metrics for survival predictions and clusters."""
import numpy as np
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import concordance_index_censored, integrated_brier_score


def eval_times(d):
    """Time grid for the IBS: 10th to 80th percentile of test times.

    Clipped to the training follow-up so the censoring KM used by IBS stays defined
    (test times can exceed max(t_tr) on small / simulated draws).
    """
    lo, hi = np.quantile(d.t_te, [0.1, 0.8])
    hi = min(float(hi), float(d.t_tr.max()) * (1 - 1e-6))
    lo = min(float(lo), hi)
    if lo >= hi:
        lo, hi = np.quantile(d.t_tr, [0.1, 0.8])
        hi = min(float(hi), float(d.t_tr.max()) * (1 - 1e-6))
    return np.linspace(lo, hi, 25)


def survival_metrics(d, risk, surv_fn=None):
    """risk: higher = earlier event. surv_fn(times) -> (n_test, n_times) survival probabilities."""
    out = {"C-index": concordance_index_censored(d.e_te.astype(bool), d.t_te, risk)[0]}
    if surv_fn is not None:
        times = eval_times(d)
        try:
            out["IBS"] = integrated_brier_score(d.y_tr, d.y_te, surv_fn(times), times)
        except ValueError:
            # sksurv needs every test time inside the training censoring KM support
            out["IBS"] = np.nan
    return out


def logrank_chi2(d, clusters):
    """Log-rank test statistic between clusters on the test set (higher = more separated)."""
    if len(np.unique(clusters)) < 2:
        return np.nan
    return multivariate_logrank_test(d.t_te, clusters, d.e_te).test_statistic
