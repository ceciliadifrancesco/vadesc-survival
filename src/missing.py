"""Missing data: simulating missingness mechanisms and a missingness-aware VaDeSC.

The model follows the idea of Collier et al. (2020), "VAEs in the presence of missing data":
the encoder sees the missingness mask, the reconstruction term only uses observed values,
and (optionally) the mask itself is modelled from the latent variable, so that informative
missingness can be captured.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vadesc import VaDeSC


# ---------- simulating missingness ----------

def _calibrate(scores, rate):
    """Offset b such that mean(sigmoid(scores + b)) == rate."""
    lo, hi = -50.0, 50.0
    for _ in range(60):
        b = (lo + hi) / 2
        if (1 / (1 + np.exp(-(scores + b)))).mean() > rate:
            hi = b
        else:
            lo = b
    return b


def add_missingness(X, mechanism, rate=0.3, strength=3.0, seed=0, event_time=None):
    """Return a copy of X with NaNs.

    MCAR: every entry is missing with the same probability.
    MAR:  the first half of the columns is always observed; whether a value in the second half is
          missing depends on an observed column.
    MNAR: self-masking; large values are more likely to be missing.
    survival: missingness depends on the (true) event time. In the first half of the columns,
          long survivors are more often missing; in the second half, short survivors are.
          Needs event_time.
    """
    rng = np.random.default_rng(seed)
    X = np.array(X, dtype=float)
    n, p = X.shape
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    if mechanism == "none":
        return X
    if mechanism == "MCAR":
        miss = rng.uniform(size=(n, p)) < rate
    elif mechanism == "MAR":
        miss = np.zeros((n, p), bool)
        observed = np.arange(p // 2)
        for j in range(p // 2, p):
            s = strength * Xs[:, rng.choice(observed)]
            # twice the rate on half the columns keeps the overall rate at `rate`
            miss[:, j] = rng.uniform(size=n) < 1 / (1 + np.exp(-(s + _calibrate(s, min(2 * rate, 0.95)))))
    elif mechanism == "MNAR":
        miss = np.zeros((n, p), bool)
        for j in range(p):
            s = strength * Xs[:, j]
            miss[:, j] = rng.uniform(size=n) < 1 / (1 + np.exp(-(s + _calibrate(s, rate))))
    elif mechanism == "survival":
        log_t = np.log(np.asarray(event_time, dtype=float) + 1e-8)
        st = (log_t - log_t.mean()) / log_t.std()
        miss = np.zeros((n, p), bool)
        for j in range(p):
            s = strength * (st if j < p // 2 else -st)
            miss[:, j] = rng.uniform(size=n) < 1 / (1 + np.exp(-(s + _calibrate(s, rate))))
    else:
        raise ValueError(mechanism)
    X[miss] = np.nan
    return X


# ---------- simulated data with missing values ----------

def simulate_splits(seed=0, mechanism="none", rate=0.3, strength=3.0, n=6000, p=30, k=3, latent_dim=8,
                    p_cens=0.3, xrange=(-0.5, 0.5), brange=(-10.0, 10.0)):
    """Simulated data (official VaDeSC generator, same settings as data.load_simulated but smaller)
    with values removed by `mechanism`, in SurvData format. Returns (data, true test clusters).

    mechanism: "none", "MCAR", "MAR", "MNAR" or "survival". Missing values are kept as NaN.
    xrange sets how far apart the cluster means are.
    """
    from .data import make_splits
    from .simulate import simulate_nonlin_profile_surv

    out = simulate_nonlin_profile_surv(p=p, n=n, k=k, latent_dim=latent_dim, p_cens=p_cens, seed=seed,
                                       balanced=True, isotropic=True, weibull_k=1, xrange=xrange,
                                       brange=brange, return_event_time=True)
    X, t, e, c, t_event = out[0], out[1], out[2].astype(int), out[3], out[-1]
    X = add_missingness(X, mechanism, rate, strength=strength, seed=seed, event_time=t_event)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    d = make_splits(X, t, e, [], f"simulated ({mechanism})", seed=seed, keep_missing=True)
    return d, c[d.idx["te"]]


# ---------- removing values from real, complete datasets ----------

# continuous measurements that are made missing; age stays observed (the MAR driver)
REAL_MISSING_COLS = {"GBSG2": ["estrec", "progrec", "pnodes", "tsize"],
                     "WHAS500": ["bmi", "diasbp", "hr", "sysbp"]}
# recorded after admission: for the 40 patients who died in hospital, los equals their survival time
REAL_DROP = {"GBSG2": [], "WHAS500": ["los"]}


def baseline_risk(X, t, e):
    """Linear predictor of a Cox model on the complete data (only used to generate missingness)."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    Z = pd.get_dummies(X, drop_first=True).astype(float)
    Z = (Z - Z.mean()) / (Z.std() + 1e-8)
    return CoxPHSurvivalAnalysis(alpha=0.1).fit(Z.to_numpy(), Surv.from_arrays(e.astype(bool), t)).predict(Z.to_numpy())


def remove_values(X, cols, mechanism, rate=0.3, strength=2.0, seed=0, driver="age", risk=None):
    """Return a copy of the DataFrame X with values of `cols` set to NaN (overall rate `rate` in those cols).

    MCAR: at random.
    MAR:  depends on an always-observed column (`driver`), e.g. older patients tested differently.
    MNAR: self-masking, larger values more likely to be missing.
    risk: depends on how sick the patient is at baseline (`risk`, e.g. a Cox linear predictor):
          the first half of `cols` is missing more often for high-risk patients, the second half
          for low-risk patients. Decided at baseline, not by the future survival time.
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
    for j, c in enumerate(cols):
        v = X[c].to_numpy(float)
        if mechanism == "none":
            continue
        if mechanism == "MCAR":
            p = np.full(len(v), rate)
        else:
            if mechanism == "MAR":
                s = strength * z(X[driver].to_numpy(float))
            elif mechanism == "MNAR":
                s = strength * z(v)
            elif mechanism == "risk":
                s = strength * z(np.asarray(risk, float)) * (1 if j < len(cols) / 2 else -1)
            else:
                raise ValueError(mechanism)
            p = 1 / (1 + np.exp(-(s + _calibrate(s, rate))))
        X.loc[rng.uniform(size=len(v)) < p, c] = np.nan
    return X


def load_real_with_missingness(name, mechanism, rate=0.3, strength=2.0, seed=0):
    """GBSG2 or WHAS500 (complete data) with values removed by `mechanism`, split like the other datasets."""
    from sksurv.datasets import load_gbsg2, load_whas500
    from .data import make_splits

    X, y = {"GBSG2": load_gbsg2, "WHAS500": load_whas500}[name]()
    X = X.drop(columns=REAL_DROP[name]).reset_index(drop=True)
    e_field, t_field = y.dtype.names
    t, e = y[t_field].astype(float), y[e_field].astype(int)
    risk = baseline_risk(X, t, e) if mechanism == "risk" else None
    X = remove_values(X, REAL_MISSING_COLS[name], mechanism, rate, strength, seed, risk=risk)
    cat = list(X.select_dtypes(exclude="number").columns)
    X[cat] = X[cat].astype(object)
    return make_splits(X, t, e, cat, f"{name} ({mechanism})", seed, keep_missing=True)


# ---------- turning NaNs into model inputs ----------

def fill_and_mask(d, with_mask):
    """Replace NaN by 0 (= the training mean, since features are standardised).

    with_mask=False: plain mean imputation.
    with_mask=True:  append the 0/1 observed-mask as extra input columns.
    """
    def f(X):
        m = (~np.isnan(X)).astype(np.float32)
        x = np.nan_to_num(X, nan=0.0).astype(np.float32)
        return np.concatenate([x, m], 1) if with_mask else x
    features = d.features + ([f"observed_{c}" for c in d.features] if with_mask else [])
    return replace(d, X_tr=f(d.X_tr), X_va=f(d.X_va), X_te=f(d.X_te), features=features)


# ---------- the model ----------

class MissVaDeSC(VaDeSC):
    """VaDeSC whose input is [x with missing values set to 0, mask].

    - the encoder sees both the values and the mask
    - the reconstruction loss only counts observed values
    - model_mask=True adds a decoder for the mask, p(m | z), so the latent space also has to
      explain which features were measured (informative missingness)
    """

    def __init__(self, d_in, d_lat=16, K=4, dist="weibull", weibull_k=2.0, survival=True, model_mask=False,
                 link="softplus", surv_weight=1.0, net="auto"):
        super().__init__(d_in, d_lat, K, dist, weibull_k, survival, link, surv_weight, net)
        self.D = d_in // 2
        self.model_mask = model_mask
        # the decoder only reconstructs the values, not the mask
        self.dec = nn.Sequential(nn.Linear(d_lat, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(),
                                 nn.Linear(50, self.D))
        self.mask_dec = nn.Sequential(nn.Linear(d_lat, 50), nn.ReLU(), nn.Linear(50, self.D))

    def _split(self, x_in):
        return x_in[:, :self.D], x_in[:, self.D:]

    def recon_loss(self, x_in, z):
        x, m = self._split(x_in)
        loss = (((x - self.dec(z)) ** 2) * m).sum(-1)
        if self.model_mask:
            loss = loss + F.binary_cross_entropy_with_logits(self.mask_dec(z), m, reduction="none").sum(-1)
        return loss

    def pretrain_loss(self, x_in):
        x, m = self._split(x_in)
        return ((((x - self.dec(self.encode(x_in)[0])) ** 2) * m).sum() / m.sum().clamp(min=1))
