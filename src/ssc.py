"""Semi-supervised clustering (Bair & Tibshirani, PLoS Biology 2004).

1. Score every feature with a univariate Cox model.
2. Keep the n_features most survival-related ones.
3. Run k-means on those features only.

Follows the official VaDeSC baseline (baselines/ssc/sscBair.py in i6092467/vadesc), with two
differences, both optional:
- rank="score" (default) ranks features by the absolute Cox score statistic, as in Bair & Tibshirani.
  The official VaDeSC code ranks by hazard ratio in descending order (rank="hazard_ratio"), which
  only keeps features whose larger values mean higher risk.
- The univariate Cox fits are replaced by the score test at beta = 0 (one closed-form pass over all
  features), which is what Bair & Tibshirani use and is much faster for 1000 features.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from .metrics import logrank_chi2


def cox_score_test(X, t, e):
    """Univariate Cox score statistic for every column of X (Breslow ties).

    Returns z = U / sqrt(V) and the one-step estimate of beta, U / V.
    """
    X = np.asarray(X, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    X, t, e = X[order], np.asarray(t)[order], np.asarray(e)[order].astype(bool)
    n = len(t)
    # risk set of patient i = everyone with t >= t_i = rows from the first occurrence of t_i onwards
    start = np.searchsorted(t, t, side="left")[e]
    S1 = np.cumsum(X[::-1], 0)[::-1]
    S2 = np.cumsum(X[::-1] ** 2, 0)[::-1]
    n_risk = (n - start)[:, None]
    mean = S1[start] / n_risk
    U = (X[e] - mean).sum(0)
    V = (S2[start] / n_risk - mean ** 2).sum(0)
    V = np.maximum(V, 1e-12)
    return U / np.sqrt(V), U / V


class SSC:
    def __init__(self, n_clusters, n_features, rank="score", seed=0):
        assert rank in ("score", "hazard_ratio")
        self.K, self.n_features, self.rank, self.seed = n_clusters, n_features, rank, seed

    def fit(self, X, t, e):
        self.X_tr = np.asarray(X)
        z, beta = cox_score_test(X, t, e)
        self.z, self.beta = z, beta
        self.order = np.argsort(-np.abs(z)) if self.rank == "score" else np.argsort(-beta)
        return self.refit(self.n_features)

    def refit(self, n_features):
        """Same feature scores, different number of features (no need to recompute the scores)."""
        self.n_features = n_features
        self.features = self.order[:n_features]
        self.km = KMeans(self.K, n_init=10, random_state=self.seed).fit(self.X_tr[:, self.features])
        return self

    def predict(self, X):
        return self.km.predict(np.asarray(X)[:, self.features])


def feature_grid(p):
    grid = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    return [g for g in grid if g < p] + [p]


def fit_ssc(d, n_clusters, how="val logrank", rank="score", seed=0, c_tr=None):
    """Fit SSC on d.X_tr and choose the number of features.

    how="val logrank": the number giving the largest log-rank statistic between clusters on the
        validation set (uses only survival data, so it can be used on real datasets).
    how="oracle": the number giving the best NMI with the true clusters on the training set, as in the
        official VaDeSC code (step 50). Uses the true labels, so it is optimistic.
    how=<int>: a fixed number of features.
    """
    p = d.X_tr.shape[1]
    if isinstance(how, int):
        return SSC(n_clusters, min(how, p), rank, seed).fit(d.X_tr, d.t_tr, d.e_tr)
    model = SSC(n_clusters, 1, rank, seed).fit(d.X_tr, d.t_tr, d.e_tr)
    if how == "val logrank":
        va = type("Val", (), {"t_te": d.t_va, "e_te": d.e_va})  # logrank_chi2 reads the test fields
        scores = {k: logrank_chi2(va, model.refit(k).predict(d.X_va)) for k in feature_grid(p)}
    elif how == "oracle":
        scores = {k: normalized_mutual_info_score(c_tr, model.refit(k).predict(d.X_tr))
                  for k in range(1, p, 50)}
    else:
        raise ValueError(how)
    best = max(scores, key=lambda k: np.nan_to_num(scores[k], nan=-1))
    model.refit(best)
    model.selection_scores = scores
    return model
