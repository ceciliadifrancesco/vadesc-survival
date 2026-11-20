"""Dataset loading and preprocessing."""
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.util import Surv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUPPORT_URL = "https://raw.githubusercontent.com/i6092467/vadesc/main/datasets/support/support2.csv"

# Same columns as dropped in the official loader (datasets/support/support_data.py)
SUPPORT_DROP = ["hospdead", "death", "prg2m", "prg6m", "dnr", "dnrday", "d.time",
                "aps", "sps", "surv2m", "surv6m", "totmcst"]
# Recorded after study entry. sfdm2 has a level "<2 mo. follow-up", which mostly means early death.
SUPPORT_POST_BASELINE = ["sfdm2", "slos", "charges", "totcst"]
SUPPORT_CAT = ["sex", "dzgroup", "dzclass", "income", "race", "ca", "sfdm2"]


@dataclass
class SurvData:
    name: str
    X_tr: np.ndarray
    X_va: np.ndarray
    X_te: np.ndarray
    t_tr: np.ndarray  # time in original units
    t_va: np.ndarray
    t_te: np.ndarray
    e_tr: np.ndarray  # 1 = event observed, 0 = censored
    e_va: np.ndarray
    e_te: np.ndarray
    features: list
    raw_te: pd.DataFrame  # unprocessed test rows, for interpreting clusters
    # True cluster labels (simulated data only); None for clinical datasets.
    clusters_true_tr: np.ndarray | None = None
    clusters_true_va: np.ndarray | None = None
    clusters_true_te: np.ndarray | None = None
    idx: dict = None  # row indices of each split in the original data

    @property
    def y_tr(self):
        return Surv.from_arrays(self.e_tr.astype(bool), self.t_tr)

    @property
    def y_te(self):
        return Surv.from_arrays(self.e_te.astype(bool), self.t_te)


def make_splits(X, t, e, cat_cols, name, seed=42, test_size=0.2, val_size=0.1, keep_missing=False,
                num_fill="median", cat_fill="most_frequent"):
    """Stratified train/val/test split. Imputation and scaling are fit on train only.

    keep_missing: leave numeric missing values as NaN (scaled, not imputed); categorical
    missing values become their own "missing" category.
    num_fill: "median" (impute, then standardise) or "mean" (standardise, then fill with 0 = the
              training mean). Ignored if keep_missing.
    cat_fill: "most_frequent" or "missing" (own category). Ignored if keep_missing.
    """
    idx = np.arange(len(t))
    idx_trva, idx_te = train_test_split(idx, test_size=test_size, stratify=e, random_state=seed)
    idx_tr, idx_va = train_test_split(idx_trva, test_size=val_size / (1 - test_size),
                                      stratify=e[idx_trva], random_state=seed)
    num_cols = [c for c in X.columns if c not in cat_cols]
    if keep_missing:
        num_pipe = StandardScaler()  # ignores NaN when fitting, keeps NaN when transforming
        cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        assert num_fill in ("median", "mean") and cat_fill in ("most_frequent", "missing")
        if num_fill == "median":
            num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        else:  # same values as keep_missing followed by filling NaN with 0
            num_pipe = make_pipeline(StandardScaler(), SimpleImputer(strategy="constant", fill_value=0.0))
        cat_imputer = (SimpleImputer(strategy="most_frequent") if cat_fill == "most_frequent"
                       else SimpleImputer(strategy="constant", fill_value="missing"))
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", make_pipeline(cat_imputer, OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat_cols),
    ]).fit(X.iloc[idx_tr])

    def transform(i):
        return pre.transform(X.iloc[i]).astype(np.float32)

    return SurvData(name, transform(idx_tr), transform(idx_va), transform(idx_te),
                    t[idx_tr], t[idx_va], t[idx_te], e[idx_tr], e[idx_va], e[idx_te],
                    [f.split("__")[1] for f in pre.get_feature_names_out()],
                    X.iloc[idx_te].reset_index(drop=True),
                    idx={"tr": idx_tr, "va": idx_va, "te": idx_te})


def load_support_raw():
    path = DATA_DIR / "support2.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SUPPORT_URL, path)
    return pd.read_csv(path, index_col=0)


def load_support(drop_post_baseline=False, seed=42, keep_missing=False, num_fill="median", cat_fill="most_frequent"):
    raw = load_support_raw()
    t = raw["d.time"].to_numpy(float)
    e = raw["death"].to_numpy(int)
    X = raw.drop(columns=SUPPORT_DROP + (SUPPORT_POST_BASELINE if drop_post_baseline else []))
    for c in ["totcst", "charges", "pafi", "sod"]:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    cat = [c for c in SUPPORT_CAT if c in X]
    X[cat] = X[cat].astype(object)
    name = "SUPPORT (baseline-only)" if drop_post_baseline else "SUPPORT"
    return make_splits(X, t, e, cat, name, seed, keep_missing=keep_missing, num_fill=num_fill, cat_fill=cat_fill)


def load_sksurv(name, seed=42, keep_missing=False):
    """FLChain, GBSG2 or WHAS500, bundled with scikit-survival. Only FLChain has missing values (creatinine)."""
    from sksurv.datasets import load_flchain, load_gbsg2, load_whas500

    X, y = {"FLChain": load_flchain, "GBSG2": load_gbsg2, "WHAS500": load_whas500}[name]()
    if name == "FLChain":
        X = X.drop(columns=["chapter"])  # cause of death, only known after the event
    event_field, time_field = y.dtype.names
    cat = list(X.select_dtypes(exclude="number").columns)
    X[cat] = X[cat].astype(object)
    return make_splits(X.reset_index(drop=True), y[time_field].astype(float),
                       y[event_field].astype(int), cat, name, seed, keep_missing=keep_missing)


def load_simulated(
    n=60000,
    p=1000,
    k=3,
    d_lat=16,
    p_cens=0.3,
    seed=42,
    test_size=0.2,
    val_size=0.1,
    **sim_kwargs,
):
    """VaDeSC nonlinear simulation, with true cluster labels on SurvData.
    """
    from .simulate import simulate_nonlin_profile_surv

    X, t, e, c, *_ = simulate_nonlin_profile_surv(
        p=p,
        n=n,
        k=k,
        latent_dim=d_lat,
        p_cens=p_cens,
        seed=seed,
        clust_mean=True,
        clust_cov=True,
        clust_coeffs=True,
        clust_intercepts=True,
        balanced=True,
        weibull_k=1,
        brange=(-10.0, 10.0),
        isotropic=True,
        xrange=(-0.5, 0.5),
        **sim_kwargs,
    )
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=int)
    c = np.asarray(c, dtype=int)

    idx = np.arange(n)
    idx_trva, idx_te = train_test_split(idx, test_size=test_size, stratify=e, random_state=seed)
    idx_tr, idx_va = train_test_split(
        idx_trva, test_size=val_size / (1 - test_size), stratify=e[idx_trva], random_state=seed
    )

    scaler = StandardScaler().fit(X[idx_tr])
    X_tr = scaler.transform(X[idx_tr]).astype(np.float32)
    X_va = scaler.transform(X[idx_va]).astype(np.float32)
    X_te = scaler.transform(X[idx_te]).astype(np.float32)
    features = [f"x{i}" for i in range(p)]
    raw_te = pd.DataFrame(X[idx_te], columns=features)
    raw_te["cluster"] = c[idx_te]

    return SurvData(
        name="sim",
        X_tr=X_tr,
        X_va=X_va,
        X_te=X_te,
        t_tr=t[idx_tr],
        t_va=t[idx_va],
        t_te=t[idx_te],
        e_tr=e[idx_tr],
        e_va=e[idx_va],
        e_te=e[idx_te],
        features=features,
        raw_te=raw_te,
        clusters_true_tr=c[idx_tr],
        clusters_true_va=c[idx_va],
        clusters_true_te=c[idx_te],
    )
