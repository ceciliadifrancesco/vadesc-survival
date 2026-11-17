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


def run_dsm(d, ks=(3, 4, 6), layers=((50,), (100,), (100, 100)), lr=1e-3, batch_size=100, n_iter=100, seed=0):
    """Deep Survival Machines (official code in src/dsm), Weibull mixture.

    Hyperparameters (number of mixture components k, hidden layers) are chosen by the validation
    negative log-likelihood, as in the auton-survival examples. Training stops early on the
    validation loss (official train_dsm). Times are scaled like for VaDeSC (t / max(t_train) + 0.001).
    Risk score for the C-index: 1 - S(t*) at t* = median training event time.
    """
    import torch
    from .dsm import losses
    from .dsm.dsm_torch import DeepSurvivalMachinesTorch
    from .dsm.utilities import train_dsm

    scale = d.t_tr.max()
    T = lambda t: torch.from_numpy(np.asarray(t, float) / scale + 1e-3).double()
    E = lambda e: torch.from_numpy(np.asarray(e, float)).double()
    X = lambda x: torch.from_numpy(np.asarray(x, float)).double()
    best = None
    for k in ks:
        for lay in layers:
            torch.manual_seed(seed)
            model = DeepSurvivalMachinesTorch(d.X_tr.shape[1], k=k, layers=list(lay), dist="Weibull", risks=1)
            model, _ = train_dsm(model, X(d.X_tr), T(d.t_tr), E(d.e_tr), X(d.X_va), T(d.t_va), E(d.e_va),
                                 n_iter=n_iter, lr=lr, elbo=True, bs=batch_size, random_seed=seed)
            with torch.no_grad():
                nll = float(losses.conditional_loss(model, X(d.X_va), T(d.t_va), E(d.e_va), elbo=False))
            if best is None or nll < best[0]:
                best = (nll, k, lay, model)
    _, k, lay, model = best

    def surv_fn(ts):
        with torch.no_grad():
            log_s = losses.predict_cdf(model, X(d.X_te), list(np.asarray(ts, float) / scale + 1e-3))
        return np.exp(np.array(log_s)).T

    t_star = np.median(d.t_tr[d.e_tr == 1])
    out = survival_metrics(d, 1 - surv_fn([t_star])[:, 0], surv_fn)
    out.update({"k": k, "layers": lay})
    return out


def run_weibull_aft(d, penalizer=0.05):
    keep = d.X_tr.std(0) > 0
    cols = [f"x{i}" for i in range(keep.sum())]
    df = pd.DataFrame(d.X_tr[:, keep], columns=cols)
    df["T"], df["E"] = np.maximum(d.t_tr, 1e-3), d.e_tr
    m = WeibullAFTFitter(penalizer=penalizer).fit(df, "T", "E")
    X_te = pd.DataFrame(d.X_te[:, keep], columns=cols)
    surv_fn = lambda ts: m.predict_survival_function(X_te, times=ts).to_numpy().T
    return survival_metrics(d, -m.predict_median(X_te).to_numpy().ravel(), surv_fn)
