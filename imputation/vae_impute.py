"""VAE imputation of SUPPORT as a function, so it can be re-run for every data split.

Same model, likelihoods, training and imputation as notebooks/04_vae_support_imputation.ipynb
(Collier et al., "VAEs in the presence of missing data"); only packaged so that the VAE is
trained on the training rows of a given split. The VAE never sees survival times or events.
"""
import math
import random
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

import src.data as team_data
from imputation.data_imputation import load_support_missing, split_indices
from src.data import load_support_raw

METHODS = ("Zero Imputation", "Encoder Indicators", "Encoder Decoder Indicators")


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class _VAE(nn.Module):
    def __init__(self, enc_in, dec_b, d, n_real, z_dim, hidden):
        super().__init__()
        self.body = _mlp((enc_in,) + hidden)
        self.mu = nn.Linear(hidden[-1], z_dim)
        self.sigma = nn.Linear(hidden[-1], z_dim)
        rev = hidden[::-1]
        self.first = nn.Sequential(nn.Linear(z_dim, rev[0]), nn.ReLU())
        self.rest = _mlp((rev[0] + dec_b,) + rev[1:]) if len(rev) > 1 else nn.Identity()
        self.out = nn.Linear(rev[-1] if len(rev) > 1 else rev[0] + dec_b, d)
        self.log_sigma = nn.Parameter(torch.zeros(n_real))

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), F.softplus(self.sigma(h)) + 1e-3

    def decode(self, z, b=None):
        h = self.first(z)
        if b is not None:
            h = torch.cat([h, b], dim=-1)
        return self.out(self.rest(h))

    def real_sigma(self):
        return self.log_sigma.clamp(-4.0, 2.0).exp()


def vae_impute_support(drop_post_baseline=True, split_seed=42, vae_seed=0, method="Encoder Indicators",
                       z_dim=16, hidden=(128, 64), lr=1e-3, batch_size=128, max_epochs=500, patience=10,
                       valid_elbo_samples=5, impute_samples=500, device="cpu", verbose=False):
    """Returns support2.csv (same rows and columns) with the gaps of the modelled variables filled in."""
    assert method in METHODS, method
    dev = torch.device(device)
    data = load_support_missing(drop_post_baseline=drop_post_baseline, seed=split_seed)
    V = [v.to_dict() for v in data.variables]
    D, N_VARS = data.X_tr.shape[1], len(V)
    real = [v for v in V if v["kind"] == "real"]
    binv = [v for v in V if v["kind"] == "binary"]
    cats = [v for v in V if v["kind"] == "categorical"]
    pos = {v["name"]: i for i, v in enumerate(V)}
    T = lambda a: torch.tensor(a, device=dev)
    REAL_COLS, BIN_COLS = T([v["start"] for v in real]), T([v["start"] for v in binv])
    REAL_VPOS, BIN_VPOS = T([pos[v["name"]] for v in real]), T([pos[v["name"]] for v in binv])
    CAT_SLICES = [(v["start"], v["stop"], pos[v["name"]]) for v in cats]
    VAR_FIRST = T([v["start"] for v in V])

    def tensors(X, M):
        x, m = torch.tensor(X, device=dev), torch.tensor(M, device=dev)
        return x * m, m

    X_tr, M_tr = tensors(data.X_tr, data.M_tr)
    X_va, M_va = tensors(data.X_va, data.M_va)
    X_te, M_te = tensors(data.X_te, data.M_te)

    def inputs(x, m):
        x0 = x * m
        if method == "Zero Imputation":
            return x0, None
        vm = m[:, VAR_FIRST]
        return torch.cat([x0, vm], -1), (vm if method == "Encoder Decoder Indicators" else None)

    def log_px(model, out, x, m):
        vm = m[..., VAR_FIRST]
        total = 0.0
        if len(real):
            ll = Normal(out[..., REAL_COLS], model.real_sigma()).log_prob(x[..., REAL_COLS])
            total = total + (ll * vm[..., REAL_VPOS]).sum(-1)
        if len(binv):
            logits = out[..., BIN_COLS]
            ll = -F.binary_cross_entropy_with_logits(logits, x[..., BIN_COLS].expand_as(logits), reduction="none")
            total = total + (ll * vm[..., BIN_VPOS]).sum(-1)
        for a, b, p in CAT_SLICES:
            total = total + (F.log_softmax(out[..., a:b], -1) * x[..., a:b]).sum(-1) * vm[..., p]
        return total

    def elbo(model, x, m, s=1):
        inp, b = inputs(x, m)
        mu, sig = model.encode(inp)
        q = Normal(mu, sig)
        z = q.rsample((s,))
        out = model.decode(z, None if b is None else b.expand(s, *b.shape))
        kl = kl_divergence(q, Normal(torch.zeros_like(mu), torch.ones_like(sig))).sum(-1)
        return log_px(model, out, x, m).mean(0) - kl

    random.seed(vae_seed); np.random.seed(vae_seed); torch.manual_seed(vae_seed)
    enc_in = D + (0 if method == "Zero Imputation" else N_VARS)
    dec_b = N_VARS if method == "Encoder Decoder Indicators" else 0
    model = _VAE(enc_in, dec_b, D, len(real), z_dim, tuple(hidden)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, wait = None, None, 0
    for epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(X_tr.shape[0], device=dev).split(batch_size):
            loss = -elbo(model, X_tr[idx], M_tr[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            opt.step()
        model.eval()
        with torch.inference_mode():
            v = elbo(model, X_va, M_va, valid_elbo_samples).mean().item()
        if best is None or v > best:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        elif wait >= patience:
            break
        else:
            wait += 1
    model.load_state_dict(best_state)
    if verbose:
        print(f"VAE ({method}, split {split_seed}): {epoch + 1} epochs, best valid ELBO {best:.3f}")

    @torch.inference_mode()
    def impute(X, M, chunk=256):
        filled = []
        for idx in torch.arange(X.shape[0], device=dev).split(chunk):
            x, m = X[idx], M[idx]
            inp, b = inputs(x, m)
            mu, sig = model.encode(inp)
            q = Normal(mu, sig)
            z = q.sample((impute_samples,))
            out = model.decode(z, None if b is None else b.expand(impute_samples, *b.shape))
            log_w = (log_px(model, out, x, m) + Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
                     - q.log_prob(z).sum(-1))
            ev = torch.empty_like(out)
            if len(real):
                ev[..., REAL_COLS] = out[..., REAL_COLS]
            if len(binv):
                ev[..., BIN_COLS] = torch.sigmoid(out[..., BIN_COLS])
            for a, c, _ in CAT_SLICES:
                ev[..., a:c] = torch.softmax(out[..., a:c], -1)
            pred = (torch.softmax(log_w, 0)[..., None] * ev).sum(0)  # importance-weighted, as in notebook 04
            if len(binv):
                pred[:, BIN_COLS] = (pred[:, BIN_COLS] >= 0.5).float()
            for a, c, _ in CAT_SLICES:
                pred[:, a:c] = F.one_hot(pred[:, a:c].argmax(-1), c - a).float()
            filled.append(m * x + (1 - m) * pred)
        return torch.cat(filled).cpu().numpy()

    torch.manual_seed(vae_seed)
    imputed = np.concatenate([impute(X_tr, M_tr), impute(X_va, M_va), impute(X_te, M_te)])

    # back to the support2.csv layout (original units, original row order)
    original = data.to_original_units(imputed)
    raw = load_support_raw()
    order = np.concatenate(split_indices(raw["death"].to_numpy(int), split_seed))
    original.index = raw.index[order]
    original = original.loc[raw.index]
    out = raw.copy()
    for c in original.columns:
        out[c] = raw[c].where(raw[c].notna(), original[c])  # observed values are kept
    return out


def load_support_from_frame(raw, drop_post_baseline=True, seed=42):
    """The team's load_support() on an (imputed) support2-like DataFrame instead of the file."""
    with mock.patch.object(team_data, "load_support_raw", lambda: raw.copy()):
        return team_data.load_support(drop_post_baseline=drop_post_baseline, seed=seed)
