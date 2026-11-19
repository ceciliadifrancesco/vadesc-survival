"""SUPPORT exactly as in the official VaDeSC code (datasets/support/support_data.py, which is based on
Chapfuwa et al.'s Survival Cluster Analysis code), and step-by-step variants between it and our loader.

Differences between the official preprocessing and ours (src/data.py, load_support):
- missing categories: official one-hot encoding (pandas.get_dummies) turns a missing category into all
  zeros; we fill it with the most frequent training category
- log transform: official abs(log(x + 1e-8)) of totcst, charges, pafi, sod (and totmcst, dropped later);
  we use log1p on totcst, charges, pafi, sod
- scaler: official fits it on ALL patients before splitting; we fit it on the training set only
- split: official shuffles once with np.random.seed(seed) and takes 80% train / 10% test / 10%
  validation, not stratified; we use 70/10/20 stratified on the event
- numeric missing values: training median in both
The post-baseline variables (sfdm2, slos, charges, totcst) are kept, as in the official code (59 inputs).

load_support_variant switches each difference on or off, so we can see which one matters.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import SUPPORT_DROP, SurvData, load_support_raw

ONE_HOT = ["sex", "dzgroup", "dzclass", "income", "race", "ca", "sfdm2"]


def load_support_variant(seed=42, missing_cat_as_zeros=True, official_log=True, scaler_on_all=True,
                         official_split=True, drop_post_baseline=False):
    """All True = the official preprocessing. All False = our preprocessing (as load_support)."""
    rng_state = np.random.RandomState(seed)
    df = load_support_raw()
    if drop_post_baseline:
        df = df.drop(columns=["sfdm2", "slos", "charges", "totcst"])
    one_hot = [c for c in ONE_HOT if c in df]
    t = df["d.time"].to_numpy(float)
    e = df["death"].to_numpy(int)

    if official_log:
        for c in ["totcst", "charges", "pafi", "sod"]:
            if c in df:
                with np.errstate(invalid="ignore", divide="ignore"):
                    df[c] = np.abs(np.log(df[c] + 1e-8))
    else:
        for c in ["totcst", "charges", "pafi", "sod"]:
            if c in df:
                df[c] = np.log1p(df[c].clip(lower=0))

    # split
    idx = np.arange(len(df))
    if official_split:
        rng_state.shuffle(idx)
        n_tr = int(0.80 * len(e))
        split = int((len(e) - n_tr) / 2)
        tr, te, va = idx[:n_tr], idx[n_tr:n_tr + split], idx[n_tr + split:]
    else:
        trva, te = train_test_split(idx, test_size=0.2, stratify=e, random_state=seed)
        tr, va = train_test_split(trva, test_size=0.1 / 0.8, stratify=e[trva], random_state=seed)

    X = df.drop(columns=[c for c in SUPPORT_DROP if c in df])
    if not missing_cat_as_zeros:  # fill missing categories with the training mode first
        for c in one_hot:
            X[c] = X[c].fillna(X[c].iloc[tr].mode()[0])
    X = pd.get_dummies(X, prefix=one_hot, columns=one_hot, dtype=float)
    is_cat = np.array([any(c.startswith(p + "_") for p in one_hot) for c in X.columns])
    x = X.to_numpy(float)

    fit_rows = np.arange(len(x)) if scaler_on_all else tr
    if official_log:  # official order: scale (ignoring NaN), then impute with the training median
        sc = StandardScaler().fit(x[fit_rows][:, ~is_cat])
        x[:, ~is_cat] = sc.transform(x[:, ~is_cat])
        x = np.where(np.isnan(x), np.nanmedian(x[tr], axis=0), x)
    else:  # our order: impute with the training median, then scale
        x = np.where(np.isnan(x), np.nanmedian(x[tr], axis=0), x)
        sc = StandardScaler().fit(x[fit_rows][:, ~is_cat])
        x[:, ~is_cat] = sc.transform(x[:, ~is_cat])
    x = x.astype(np.float32)

    name = "SUPPORT (" + ("official" if all([missing_cat_as_zeros, official_log, scaler_on_all, official_split])
                          else "variant") + " preprocessing)"
    return SurvData(name, x[tr], x[va], x[te], t[tr], t[va], t[te], e[tr], e[va], e[te],
                    list(X.columns), pd.DataFrame(), idx={"tr": tr, "va": va, "te": te})


def load_support_paper(seed=42):
    """The official preprocessing and split."""
    return load_support_variant(seed)
