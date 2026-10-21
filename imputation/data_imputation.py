"""SUPPORT loading for missing-data / VAE imputation experiments.

Builds on the team's src/data.py without changing it: the raw file, the dropped
columns and the train/valid/test split are taken from there, so patients end up
in the same split as with load_support().

    load_support_missing(drop_post_baseline=False, seed=42)
        Encoded SUPPORT with the gaps kept and the real missingness mask.
    load_support_imputed(path, drop_post_baseline=False)
        Reads a VAE-imputed *_original_units.csv back into a SurvData object,
        preprocessed exactly like load_support(), for the survival models.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.util import Surv

# make `src` importable when this file is used from its own folder
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from src.data import (SUPPORT_CAT, SUPPORT_DROP, SUPPORT_POST_BASELINE,  # noqa: E402
                      SurvData, load_support_raw)

# Binary variables, encoded as a single 0/1 column: {column: value that maps to 1}
SUPPORT_BINARY = {"sex": "male", "diabetes": 1, "dementia": 1}
# Multi-level categoricals, one-hot encoded as a group
SUPPORT_MULTICAT = ["dzgroup", "dzclass", "income", "race", "ca", "sfdm2"]
# Strongly right-skewed variables, log1p-transformed before standardizing.
# Not urine (93 zeros make log1p worse) and not sod / pafi (barely skewed).
SUPPORT_LOG = ["totcst", "charges", "bili", "crea", "bun", "wblc", "glucose"]


def split_indices(e, seed=42, test_size=0.2, val_size=0.1):
    """Stratified train/val/test indices; identical to the split in src.data.make_splits."""
    idx = np.arange(len(e))
    idx_trva, idx_te = train_test_split(idx, test_size=test_size, stratify=e, random_state=seed)
    idx_tr, idx_va = train_test_split(idx_trva, test_size=val_size / (1 - test_size),
                                      stratify=e[idx_trva], random_state=seed)
    return idx_tr, idx_va, idx_te


@dataclass
class Variable:
    """One original column and where it lives in the encoded matrix."""
    name: str
    kind: str          # "real", "binary" or "categorical"
    cols: slice        # columns of X / M belonging to this variable
    log: bool = False  # real: log1p applied before standardizing
    mean: float = 0.0  # real: training mean / std (after log, if any)
    std: float = 1.0
    positive: object = None  # binary: original value encoded as 1
    categories: list = field(default_factory=list)  # categorical: one column each

    def to_dict(self):
        """JSON-friendly description (used for support_vae_variables.json)."""
        return {"name": self.name, "kind": self.kind, "start": self.cols.start,
                "stop": self.cols.stop, "log": self.log, "mean": self.mean, "std": self.std,
                "positive": self.positive, "categories": self.categories}


@dataclass
class MissingSurvData:
    """Encoded SUPPORT data with the real missingness mask.

    X_*: float32 [n, d]. Real variables standardized, binaries 0/1, categoricals
         one-hot. Missing entries are set to 0 (= training mean for real variables).
    M_*: float32 [n, d]. 1 = observed, 0 = missing. All columns of one categorical
         variable share the same mask value.
    """
    name: str
    X_tr: np.ndarray
    X_va: np.ndarray
    X_te: np.ndarray
    M_tr: np.ndarray
    M_va: np.ndarray
    M_te: np.ndarray
    t_tr: np.ndarray
    t_va: np.ndarray
    t_te: np.ndarray
    e_tr: np.ndarray
    e_va: np.ndarray
    e_te: np.ndarray
    variables: list    # list[Variable], in column order
    features: list     # column names of X
    raw_te: pd.DataFrame

    @property
    def y_tr(self):
        return Surv.from_arrays(self.e_tr.astype(bool), self.t_tr)

    @property
    def y_te(self):
        return Surv.from_arrays(self.e_te.astype(bool), self.t_te)

    def to_original_units(self, X):
        """Decode an encoded (e.g. imputed) matrix back to a DataFrame of original values.

        Binaries are thresholded at 0.5 and categoricals take the argmax column.
        """
        out = {}
        for v in self.variables:
            block = X[:, v.cols]
            if v.kind == "real":
                x = block[:, 0] * v.std + v.mean
                out[v.name] = np.expm1(x) if v.log else x
            elif v.kind == "binary":
                neg = _other_binary_value(v.positive)
                out[v.name] = np.where(block[:, 0] >= 0.5, v.positive, neg)
            else:
                out[v.name] = np.asarray(v.categories, dtype=object)[block.argmax(1)]
        return pd.DataFrame(out)


def _other_binary_value(positive):
    return {"male": "female", 1: 0}.get(positive, f"not {positive}")


def _encode_support(X, idx_tr):
    """Fits encodings on the training rows and returns (X_all, M_all, variables, features)."""
    blocks_x, blocks_m, variables, features = [], [], [], []
    col = 0
    tr = X.iloc[idx_tr]

    for name in X.columns:
        values = X[name]
        observed = values.notna().to_numpy()

        if name in SUPPORT_BINARY:
            pos = SUPPORT_BINARY[name]
            x = (values == pos).to_numpy(np.float32)[:, None]
            m = observed.astype(np.float32)[:, None]
            var = Variable(name, "binary", slice(col, col + 1), positive=pos)
            names = [name if pos == 1 else f"{name}_{pos}"]

        elif name in SUPPORT_MULTICAT:
            cats = sorted(tr[name].dropna().unique().tolist())
            onehot = pd.get_dummies(pd.Categorical(values, categories=cats)).to_numpy(np.float32)
            # values not in the training categories also count as missing
            known = observed & values.isin(cats).to_numpy()
            x = onehot * known[:, None]
            m = np.repeat(known.astype(np.float32)[:, None], len(cats), axis=1)
            var = Variable(name, "categorical", slice(col, col + len(cats)), categories=cats)
            names = [f"{name}_{c}" for c in cats]

        else:  # real-valued
            log = name in SUPPORT_LOG
            v = values.astype(float)
            if log:
                v = np.log1p(v.clip(lower=0))
            mean = float(v.iloc[idx_tr].mean())       # pandas ignores NaN
            std = float(v.iloc[idx_tr].std(ddof=0)) or 1.0
            x = ((v - mean) / std).fillna(0.0).to_numpy(np.float32)[:, None]
            m = observed.astype(np.float32)[:, None]
            var = Variable(name, "real", slice(col, col + 1), log=log, mean=mean, std=std)
            names = [name]

        x[m == 0] = 0.0
        blocks_x.append(x)
        blocks_m.append(m)
        variables.append(var)
        features.extend(names)
        col += x.shape[1]

    return np.hstack(blocks_x), np.hstack(blocks_m), variables, features


def load_support_missing(drop_post_baseline=False, seed=42):
    """SUPPORT encoded for a missing-data VAE: data + real missingness mask.

    Same rows, dropped columns and train/val/test split as src.data.load_support
    (for the same seed), but nothing is imputed. Encodings (means, stds,
    categories) are fit on the training split only.
    """
    raw = load_support_raw()
    t = raw["d.time"].to_numpy(float)
    e = raw["death"].to_numpy(int)
    X = raw.drop(columns=SUPPORT_DROP + (SUPPORT_POST_BASELINE if drop_post_baseline else []))
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].str.strip()

    idx_tr, idx_va, idx_te = split_indices(e, seed)
    X_all, M_all, variables, features = _encode_support(X, idx_tr)

    name = "SUPPORT (baseline-only)" if drop_post_baseline else "SUPPORT"
    return MissingSurvData(
        name,
        X_all[idx_tr], X_all[idx_va], X_all[idx_te],
        M_all[idx_tr], M_all[idx_va], M_all[idx_te],
        t[idx_tr], t[idx_va], t[idx_te], e[idx_tr], e[idx_va], e[idx_te],
        variables, features, X.iloc[idx_te].reset_index(drop=True))


def load_support_imputed(path, drop_post_baseline=False):
    """SurvData from a VAE-imputed *_original_units.csv file.

    Uses the CSV's own train/valid/test split and the same preprocessing as
    src.data.load_support, so only the imputation differs from the baseline pipeline.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["split", "time", "event"])
    if drop_post_baseline:
        X = X.drop(columns=[c for c in SUPPORT_POST_BASELINE if c in X])
    for c in ["totcst", "charges", "pafi", "sod"]:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    cat = [c for c in SUPPORT_CAT if c in X]
    X[cat] = X[cat].astype(object)
    t, e = df["time"].to_numpy(float), df["event"].to_numpy(int)
    idx = {s: np.flatnonzero(df["split"].to_numpy() == s) for s in ("train", "valid", "test")}

    num_cols = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
    ]).fit(X.iloc[idx["train"]])

    def transform(i):
        return pre.transform(X.iloc[i]).astype(np.float32)

    tr, va, te = idx["train"], idx["valid"], idx["test"]
    return SurvData("SUPPORT (VAE-imputed)", transform(tr), transform(va), transform(te),
                    t[tr], t[va], t[te], e[tr], e[va], e[te],
                    [f.split("__")[1] for f in pre.get_feature_names_out()],
                    X.iloc[te].reset_index(drop=True))