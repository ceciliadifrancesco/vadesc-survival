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
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class VaDeSC(nn.Module):
    def __init__(self, d_in, d_lat=16, K=4, dist="weibull", weibull_k=2.0, survival=True,
                 link="softplus", surv_weight=1.0, net="auto"):
        super().__init__()
        self.K, self.dist, self.k, self.survival = K, dist, weibull_k, survival
        self.link, self.surv_weight = link, surv_weight  # defaults = official code
        self.learn_k = False  # set True to learn the Weibull shape instead of fixing it
        # Match official networks.py: Encoder_small/Decoder_small if d_in <= 100
        # (SUPPORT), else wide Encoder/Decoder (sim, p=1000).
        # net="small" / "wide" forces one size, e.g. to compare models whose inputs straddle 100.
        assert net in ("auto", "small", "wide")
        if net == "small" or (net == "auto" and d_in <= 100):
            self.enc = nn.Sequential(nn.Linear(d_in, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU())
            self.enc_mu = nn.Linear(100, d_lat)
            self.enc_logvar = nn.Linear(100, d_lat)
            self.dec = nn.Sequential(nn.Linear(d_lat, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(),
                                     nn.Linear(50, d_in))
        else:
            self.enc = nn.Sequential(
                nn.Linear(d_in, 500), nn.ReLU(),
                nn.Linear(500, 500), nn.ReLU(),
                nn.Linear(500, 2000), nn.ReLU(),
            )
            self.enc_mu = nn.Linear(2000, d_lat)
            self.enc_logvar = nn.Linear(2000, d_lat)
            self.dec = nn.Sequential(
                nn.Linear(d_lat, 2000), nn.ReLU(),
                nn.Linear(2000, 500), nn.ReLU(),
                nn.Linear(500, 500), nn.ReLU(),
                nn.Linear(500, d_in),
            )
        # GMM prior p(c) p(z|c)
        self.prior_logits = nn.Parameter(torch.zeros(K))
        self.c_mu = nn.Parameter(torch.randn(K, d_lat) * 0.1)
        self.c_logvar = nn.Parameter(torch.zeros(K, d_lat))
        # cluster-specific survival heads, linear in z
        self.beta = nn.Parameter(torch.randn(K, d_lat) * 0.1)
        self.beta0 = nn.Parameter(torch.zeros(K))
        self.log_sigma = nn.Parameter(torch.zeros(K))  # log-normal variant only
        self.log_k = nn.Parameter(torch.tensor(math.log(weibull_k)))  # used only if learn_k

    def encode(self, x):
        h = self.enc(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def log_prior(self):
        return F.log_softmax(self.prior_logits, 0)

    def log_p_z_c(self, z):
        """(B, K) log N(z | mu_c, sigma_c)"""
        z = z.unsqueeze(1)
        return -0.5 * (LOG2PI + self.c_logvar + (z - self.c_mu) ** 2 / self.c_logvar.exp()).sum(-1)

    def weibull_scale(self, eta):
        """Weibull scale from the linear predictor. The official code uses softplus; link="exp"
        makes covariates act multiplicatively on time (log-linear), like an AFT / Cox model."""
        if self.link == "exp":
            return torch.exp(eta.clamp(-20, 20))
        return F.softplus(eta) + 1e-6

    def shape(self):
        return self.log_k.exp() if self.learn_k else self.k

    def log_lik_t(self, z, t, d):
        """(B, K) right-censored log-likelihood of (t, d) under each cluster's survival model."""
        eta = z @ self.beta.T + self.beta0
        t, d = t.unsqueeze(1), d.unsqueeze(1)
        if self.dist == "weibull":
            lam = self.weibull_scale(eta)
            k = self.shape()
            log_hazard = math.log(k) if not self.learn_k else torch.log(k)
            log_hazard = log_hazard - torch.log(lam) + (k - 1) * (torch.log(t) - torch.log(lam))
            return d * log_hazard - (t / lam) ** k  # = d log f + (1 - d) log S
        sigma = self.log_sigma.exp()
        u = (torch.log(t) - eta) / sigma
        log_f = -0.5 * (LOG2PI + u ** 2) - torch.log(sigma) - torch.log(t)
        return d * log_f + (1 - d) * torch.special.log_ndtr(-u)

    def recon_loss(self, x, z):
        """(B,) negative log-likelihood of x given z, up to constants."""
        return ((x - self.dec(z)) ** 2).sum(-1)

    def pretrain_loss(self, x):
        return F.mse_loss(self.dec(self.encode(x)[0]), x)

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
        loss = (self.recon_loss(x, z)
                - (gamma * log_pzc).sum(-1)
                - (gamma * self.log_prior()).sum(-1)
                + (gamma * torch.log(gamma + 1e-10)).sum(-1)
                - 0.5 * (logvar + 1).sum(-1))
        if self.survival:
            loss = loss - self.surv_weight * (gamma * ll_t).sum(-1)
        return loss.mean()

    @torch.no_grad()
    def predict(self, x):
        """Latent means and p(c | z). Survival time is unknown at test time (paper: w/o t)."""
        mu, _ = self.encode(x)
        return mu, (self.log_prior() + self.log_p_z_c(mu)).softmax(-1)

    @torch.no_grad()
    def assign_with_t(self, x, t, d):
        """Cluster probabilities q(c | z, t), using the survival time. Only for evaluating clustering
        on simulated data (as in the paper's "with t" results), never for prediction."""
        mu, _ = self.encode(x)
        return (self.log_prior() + self.log_p_z_c(mu) + self.log_lik_t(mu, t, d)).softmax(-1)

    @torch.no_grad()
    def survival_curves(self, x, times):
        """Mixture survival function S(t | x) at the given times (model time scale)."""
        mu, gamma = self.predict(x)
        eta = (mu @ self.beta.T + self.beta0).unsqueeze(-1)
        tt = times.view(1, 1, -1)
        if self.dist == "weibull":
            S = torch.exp(-(tt / self.weibull_scale(eta)) ** self.shape())
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
        scale = self.weibull_scale(eta) if self.dist == "weibull" else eta
        return -scale.gather(1, gamma.argmax(1, keepdim=True)).squeeze(1)


def tensor(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


def train_vadesc(d, K=4, d_lat=16, dist="weibull", weibull_k=2.0, survival=True, epochs=300,
                 pretrain_epochs=10, lr=1e-3, batch_size=256, seed=42, verbose=False,
                 model_cls=VaDeSC, model_kwargs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = min(batch_size, max(32, len(d.t_tr) // 10))  # keep ~10 steps per epoch on small data
    t_scale = d.t_tr.max()  # official code: t / max(t) + 0.001
    X, T, E = tensor(d.X_tr), tensor(d.t_tr / t_scale + 1e-3), tensor(d.e_tr)
    model = model_cls(X.shape[1], d_lat, K, dist, weibull_k, survival, **(model_kwargs or {})).to(DEVICE)
    model.t_scale = t_scale
    n = len(X)

    def batches():
        return torch.randperm(n, device=DEVICE).split(batch_size)

    # 1) pretrain a plain autoencoder
    ae_params = [*model.enc.parameters(), *model.enc_mu.parameters(), *model.dec.parameters()]
    opt = torch.optim.Adam(ae_params, lr=1e-3)
    for _ in range(pretrain_epochs):
        for idx in batches():
            loss = model.pretrain_loss(X[idx])
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
            row = {"epoch": ep, "train_loss": total / n, "val_C": c_va}
            if getattr(d, "clusters_true_te", None) is not None:
                from sklearn.metrics import adjusted_rand_score
                _, gamma = model.predict(tensor(d.X_te))
                row["ARI"] = adjusted_rand_score(d.clusters_true_te, gamma.argmax(1).cpu().numpy())
            history.append(row)
            if verbose:
                msg = f"epoch {ep:4d} | -ELBO {total / n:9.3f} | val C-index {c_va:.3f}"
                if "ARI" in row:
                    msg += f" | ARI {row['ARI']:.3f}"
                print(msg, flush=True)
    model.history = pd.DataFrame(history)
    return model


def evaluate_vadesc(model, d):
    """Returns (metrics, test-set cluster labels, test-set latent means).

    Cluster labels use p(c | z) only (paper: Test w/o t). When true labels exist,
    also adds ARI vs ``d.clusters_true_te``.
    """
    model.eval()
    X = tensor(d.X_te)
    risk = model.risk_score(X).cpu().numpy()
    surv_fn = lambda ts: model.survival_curves(X, tensor(ts / model.t_scale + 1e-3)).cpu().numpy()
    metrics = survival_metrics(d, risk, surv_fn) if model.survival else {}
    z, gamma = model.predict(X)
    clusters = gamma.argmax(1).cpu().numpy()
    metrics["log-rank chi2"] = logrank_chi2(d, clusters)
    if getattr(d, "clusters_true_te", None) is not None:
        from sklearn.metrics import adjusted_rand_score
        metrics["ARI"] = adjusted_rand_score(d.clusters_true_te, clusters)
    return metrics, clusters, z.cpu().numpy()
