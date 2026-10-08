"""VaDeSC (Manduchi et al., ICLR 2022) in PyTorch.

Follows the official TensorFlow implementation (models/model.py in i6092467/vadesc):
same network sizes, ELBO terms, pretraining and GMM initialisation.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sksurv.metrics import concordance_index_censored

from .metrics import logrank_chi2, survival_metrics

LOG2PI = math.log(2 * math.pi)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VaDeSC(nn.Module):
    def __init__(self, d_in, d_lat=16, K=4, dist="weibull", weibull_k=2.0, survival=True):
        super().__init__()
        self.K, self.dist, self.k, self.survival = K, dist, weibull_k, survival
        self.enc = nn.Sequential(nn.Linear(d_in, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU())
        self.enc_mu = nn.Linear(100, d_lat)
        self.enc_logvar = nn.Linear(100, d_lat)
        self.dec = nn.Sequential(nn.Linear(d_lat, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(),
                                 nn.Linear(50, d_in))
        # GMM prior p(c) p(z|c)
        self.prior_logits = nn.Parameter(torch.zeros(K))
        self.c_mu = nn.Parameter(torch.randn(K, d_lat) * 0.1)
        self.c_logvar = nn.Parameter(torch.zeros(K, d_lat))
        # cluster-specific survival heads, linear in z
        self.beta = nn.Parameter(torch.randn(K, d_lat) * 0.1)
        self.beta0 = nn.Parameter(torch.zeros(K))
        self.log_sigma = nn.Parameter(torch.zeros(K))  # log-normal variant only

    def encode(self, x):
        h = self.enc(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def log_prior(self):
        return F.log_softmax(self.prior_logits, 0)

    def log_p_z_c(self, z):
        """(B, K) log N(z | mu_c, sigma_c)"""
        z = z.unsqueeze(1)
        return -0.5 * (LOG2PI + self.c_logvar + (z - self.c_mu) ** 2 / self.c_logvar.exp()).sum(-1)

    def log_lik_t(self, z, t, d):
        """(B, K) right-censored log-likelihood of (t, d) under each cluster's survival model."""
        eta = z @ self.beta.T + self.beta0
        t, d = t.unsqueeze(1), d.unsqueeze(1)
        if self.dist == "weibull":
            lam = F.softplus(eta) + 1e-6
            log_hazard = math.log(self.k) - torch.log(lam) + (self.k - 1) * (torch.log(t) - torch.log(lam))
            return d * log_hazard - (t / lam) ** self.k  # = d log f + (1 - d) log S
        sigma = self.log_sigma.exp()
        u = (torch.log(t) - eta) / sigma
        log_f = -0.5 * (LOG2PI + u ** 2) - torch.log(sigma) - torch.log(t)
        return d * log_f + (1 - d) * torch.special.log_ndtr(-u)

    def forward(self, x, t, d):
        """Negative ELBO, averaged over the batch."""
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        log_pzc = self.log_p_z_c(z)
        logits = self.log_prior() + log_pzc
        if self.survival:
            ll_t = self.log_lik_t(mu, t, d)  # the official code uses the posterior mean here
            logits = logits + ll_t
        gamma = logits.softmax(-1)  # q(c | z, t)
        loss = (((x - self.dec(z)) ** 2).sum(-1)
                - (gamma * log_pzc).sum(-1)
                - (gamma * self.log_prior()).sum(-1)
                + (gamma * torch.log(gamma + 1e-10)).sum(-1)
                - 0.5 * (logvar + 1).sum(-1))
        if self.survival:
            loss = loss - (gamma * ll_t).sum(-1)
        return loss.mean()

    @torch.no_grad()
    def predict(self, x):
        """Latent means and p(c | z). Survival time is unknown at test time."""
        mu, _ = self.encode(x)
        return mu, (self.log_prior() + self.log_p_z_c(mu)).softmax(-1)

    @torch.no_grad()
    def survival_curves(self, x, times):
        """Mixture survival function S(t | x) at the given times (model time scale)."""
        mu, gamma = self.predict(x)
        eta = (mu @ self.beta.T + self.beta0).unsqueeze(-1)
        tt = times.view(1, 1, -1)
        if self.dist == "weibull":
            S = torch.exp(-(tt / (F.softplus(eta) + 1e-6)) ** self.k)
        else:
            S = torch.special.ndtr(-(torch.log(tt) - eta) / self.log_sigma.exp().view(1, -1, 1))
        return (gamma.unsqueeze(-1) * S).sum(1)

    @torch.no_grad()
    def risk_score(self, x):
        """As in the official code: negative time scale of the most likely cluster.

        A soft mixture of per-cluster medians ranks worse, because clusters with
        very large scales dominate it.
        """
        mu, gamma = self.predict(x)
        eta = mu @ self.beta.T + self.beta0
        scale = F.softplus(eta) if self.dist == "weibull" else eta
        return -scale.gather(1, gamma.argmax(1, keepdim=True)).squeeze(1)


def tensor(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


def train_vadesc(d, K=4, d_lat=16, dist="weibull", weibull_k=2.0, survival=True, epochs=300,
                 pretrain_epochs=10, lr=1e-3, batch_size=256, seed=42, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = min(batch_size, max(32, len(d.t_tr) // 10))  # keep ~10 steps per epoch on small data
    t_scale = d.t_tr.max()  # official code: t / max(t) + 0.001
    X, T, E = tensor(d.X_tr), tensor(d.t_tr / t_scale + 1e-3), tensor(d.e_tr)
    model = VaDeSC(X.shape[1], d_lat, K, dist, weibull_k, survival).to(DEVICE)
    model.t_scale = t_scale
    n = len(X)

    def batches():
        return torch.randperm(n, device=DEVICE).split(batch_size)

    # 1) pretrain a plain autoencoder
    ae_params = [*model.enc.parameters(), *model.enc_mu.parameters(), *model.dec.parameters()]
    opt = torch.optim.Adam(ae_params, lr=1e-3)
    for _ in range(pretrain_epochs):
        for idx in batches():
            loss = F.mse_loss(model.dec(model.encode(X[idx])[0]), X[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 2) initialise the GMM prior from a GMM fit on the embeddings
    Z = model.encode(X)[0].detach().cpu().numpy()
    gmm = GaussianMixture(K, covariance_type="diag", n_init=3, random_state=seed).fit(Z)
    with torch.no_grad():
        model.c_mu.copy_(tensor(gmm.means_))
        model.c_logvar.copy_(tensor(np.log(gmm.covariances_)))
        model.prior_logits.copy_(tensor(np.log(gmm.weights_)))

    # 3) maximise the ELBO
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for idx in batches():
            loss = model(X[idx], T[idx], E[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 25 == 0 or ep == epochs:
            model.eval()
            risk_va = model.risk_score(tensor(d.X_va)).cpu().numpy()
            c_va = concordance_index_censored(d.e_va.astype(bool), d.t_va, risk_va)[0]
            history.append({"epoch": ep, "train_loss": total / n, "val_C": c_va})
            if verbose:
                print(f"epoch {ep:4d} | -ELBO {total / n:9.3f} | val C-index {c_va:.3f}")
    model.history = pd.DataFrame(history)
    return model


def evaluate_vadesc(model, d):
    """Returns (metrics, test-set cluster labels, test-set latent means)."""
    model.eval()
    X = tensor(d.X_te)
    risk = model.risk_score(X).cpu().numpy()
    surv_fn = lambda ts: model.survival_curves(X, tensor(ts / model.t_scale + 1e-3)).cpu().numpy()
    metrics = survival_metrics(d, risk, surv_fn) if model.survival else {}
    z, gamma = model.predict(X)
    clusters = gamma.argmax(1).cpu().numpy()
    metrics["log-rank chi2"] = logrank_chi2(d, clusters)
    return metrics, clusters, z.cpu().numpy()
