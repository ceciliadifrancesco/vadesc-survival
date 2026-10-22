"""VaDeSC nonlinear survival simulation (Manduchi et al.).

Ported from the official repo
  https://github.com/i6092467/vadesc
  datasets/simulations.py  (simulate_nonlin_profile_surv)
  utils/sim_utils.py       (random_nonlin_map, ReLU)
"""
import numpy as np
from numpy.random import choice, multivariate_normal, uniform
from scipy.stats import weibull_min
from sklearn.datasets import make_low_rank_matrix, make_spd_matrix


def ReLU(x):
    return x * (x > 0)


def random_nonlin_map(n_in, n_out, n_hidden, rank=1000):
    """Random MLP decoder: latents -> observed features (utils/sim_utils.py)."""
    W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
    W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
    W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
    b_0 = np.random.uniform(0, 0, (1, n_hidden))
    b_1 = np.random.uniform(0, 0, (1, n_hidden))
    b_2 = np.random.uniform(0, 0, (1, n_out))

    nlin_map = lambda x: np.matmul(ReLU(np.matmul(ReLU(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))),
                                                       W_1) + np.tile(b_1, (x.shape[0], 1))), W_2) + \
                         np.tile(b_2, (x.shape[0], 1))

    return nlin_map


def simulate_nonlin_profile_surv(p: int, n: int, k: int, latent_dim: int, p_cens: float, seed: int, p_c=None,
                                 balanced=False, clust_mean=True, clust_cov=True, isotropic=False, clust_coeffs=True,
                                 clust_intercepts=True, weibull_k=1, xrange=(-5, 5), brange=(-1, 1)):
    """Heterogeneous survival profiles with nonlinear covariates (datasets/simulations.py).

    Covariates are generated from latent features using an MLP decoder.
    Returns X, t, d, c, Z, mlp_dec, means, cov_mats, coeffs, intercepts
    (d == 1 if event, 0 if censored; c = true cluster labels).
    """
    np.random.seed(seed)

    assert p > 0 and latent_dim > 0 and n > 0 and k > 0
    assert 1 < k < n
    assert latent_dim < p
    assert len(xrange) == 2 and xrange[0] < xrange[1]
    assert len(brange) == 2 and brange[0] < brange[1]
    assert weibull_k > 0

    # Cluster prior probabilities and assignments
    if p_c is not None:
        assert len(p_c) == k and sum(p_c) == 1
    else:
        if balanced:
            p_c = np.ones((k, )) / k
        else:
            p_c = uniform(0, 1, (k, ))
            p_c = p_c / np.sum(p_c)
    c = choice(a=np.arange(k), size=(n, ), replace=True, p=p_c)

    # Cluster-specific latent means
    means = np.zeros((k, latent_dim))
    mu = uniform(xrange[0], xrange[1], (1, latent_dim))
    for l in range(k):
        if clust_mean:
            mu_l = uniform(xrange[0], xrange[1], (1, latent_dim))
            means[l, :] = mu_l
        else:
            means[l, :] = mu

    # Cluster-specific latent covariances
    cov_mats = []
    sigma = make_spd_matrix(latent_dim, random_state=seed)
    if isotropic:
        sigma = sigma * np.eye(latent_dim)
    for l in range(k):
        if clust_cov:
            sigma_l = make_spd_matrix(latent_dim, random_state=(seed + l))
            if isotropic:
                sigma_l = sigma_l * np.eye(latent_dim)
            cov_mats.append(sigma_l)
        else:
            cov_mats.append(sigma)

    # Latent features Z, then nonlinear predictors X = MLP(Z)
    Z = np.zeros((n, latent_dim))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = multivariate_normal(mean=means[l, :], cov=cov_mats[l], size=n_l)
        Z[c == l, :] = Z_l
    mlp_dec = random_nonlin_map(n_in=latent_dim, n_out=p, n_hidden=int((latent_dim + p) / 2))
    X = mlp_dec(Z)

    # Cluster-specific Weibull survival coefficients (on Z, not X)
    coeffs = np.zeros((k, latent_dim))
    intercepts = np.zeros((k, ))
    beta = uniform(brange[0], brange[1], (1, latent_dim))
    beta0 = uniform(brange[0], brange[1], (1, 1))
    for l in range(k):
        if clust_coeffs:
            beta_l = uniform(brange[0], brange[1], (1, latent_dim))
            coeffs[l, :] = beta_l
        else:
            coeffs[l, :] = beta
        if clust_intercepts:
            beta0_l = uniform(brange[0], brange[1], (1, 1))
            intercepts[l] = beta0_l.item()
        else:
            intercepts[l] = np.asarray(beta0).item()

    # Survival times
    t = np.zeros((n, ))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = Z[c == l, :]
        coeffs_l = np.expand_dims(coeffs[l, :], 1)
        intercept_l = intercepts[l]
        logexps_l = np.log(1 + np.exp(intercept_l + np.squeeze(np.matmul(Z_l, coeffs_l))))
        t_l = weibull_min.rvs(weibull_k, loc=0, scale=logexps_l, size=n_l)
        t[c == l] = t_l

    # Censoring (d == 1 if event; 0 if censored)
    d = (uniform(0, 1, (n, )) >= p_cens) * 1.0
    t_cens = uniform(0, t, (n, ))
    t[d == 0] = t_cens[d == 0]

    return X, t, d, c, Z, mlp_dec, means, cov_mats, coeffs, intercepts
