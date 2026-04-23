"""Упрощённая PyTorch-реализация TimeGAN (Yoon et al., NeurIPS 2019).

Модули:
    Embedder   e: X (B,T,d)        -> H (B,T,h)
    Recovery   r: H (B,T,h)        -> X~ (B,T,d)
    Generator  g: Z (B,T,h)        -> E_hat (B,T,h)
    Supervisor s: E_hat (B,T,h)    -> E_next (B,T,h)
    Discrim.   d: H (B,T,h)        -> y (B,T,1)

Тренировка (как в оригинальной статье, BCE-GAN):
    Phase 1 (Autoencoder): minimize MSE( X, r(e(X)) )
    Phase 2 (Supervisor) : minimize MSE( e(X)[1:], s(e(X))[:-1] )
    Phase 3 (Joint):
        Generator-side:
            L_G = BCE( d(s(g(Z))), 1 )
                  + 100 * MSE( e(X)[1:], s(e(X))[:-1] )
                  + gamma_moment * ( |Mean(X) - Mean(X_hat)| + |Std(X) - Std(X_hat)| )
                  + 10  * MSE( X, r(e(X)) )
        Discriminator-side:
            L_D = BCE( d(e(X)), 1 ) + BCE( d(s(g(Z))), 0 )
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------- утилиты для подготовки окон ----------

def make_windows(series: np.ndarray, T: int, stride: int = 1) -> np.ndarray:
    if series.ndim == 1:
        series = series[:, None]
    N, d = series.shape
    if N < T:
        raise ValueError(f"series length {N} < window {T}")
    idx = np.arange(0, N - T + 1, stride)
    out = np.stack([series[i : i + T] for i in idx], axis=0)
    return out.astype(np.float32)


def fit_minmax(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mn = x.min(axis=(0, 1), keepdims=False)
    mx = x.max(axis=(0, 1), keepdims=False)
    return mn, mx


def apply_minmax(x: np.ndarray, mn, mx, eps: float = 1e-7) -> np.ndarray:
    return (x - mn) / (mx - mn + eps)


def invert_minmax(x: np.ndarray, mn, mx, eps: float = 1e-7) -> np.ndarray:
    return x * (mx - mn + eps) + mn


# ---------- модули ----------

class _GRU(nn.Module):
    def __init__(self, in_dim, hid, out_dim, num_layers=2, out_act=nn.Sigmoid):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hid, num_layers, batch_first=True)
        self.fc = nn.Linear(hid, out_dim)
        self.act = out_act() if out_act is not None else nn.Identity()

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.act(self.fc(h))


class Embedder(_GRU):
    def __init__(self, d, h, layers=2):
        super().__init__(d, h, h, layers, out_act=nn.Sigmoid)


class Recovery(_GRU):
    def __init__(self, h, d, layers=2):
        super().__init__(h, h, d, layers, out_act=nn.Sigmoid)


class Generator(_GRU):
    def __init__(self, h, layers=2):
        super().__init__(h, h, h, layers, out_act=nn.Sigmoid)


class Supervisor(_GRU):
    def __init__(self, h, layers=1):
        super().__init__(h, h, h, layers, out_act=nn.Sigmoid)


class Discriminator(_GRU):
    """Возвращает logits (без активации) — BCE с logits."""
    def __init__(self, h, layers=2):
        super().__init__(h, h, 1, layers, out_act=None)


# ---------- основная модель ----------

@dataclass
class TimeGANConfig:
    seq_len: int = 24
    hidden_dim: int = 24
    num_layers: int = 2
    batch_size: int = 128
    ae_epochs: int = 400
    sup_epochs: int = 300
    joint_epochs: int = 400
    lr: float = 1e-3
    gamma_moment: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TimeGAN:
    def __init__(self, d: int, cfg: TimeGANConfig):
        self.d = d
        self.cfg = cfg
        h = cfg.hidden_dim
        self.embedder = Embedder(d, h, cfg.num_layers).to(cfg.device)
        self.recovery = Recovery(h, d, cfg.num_layers).to(cfg.device)
        self.generator = Generator(h, cfg.num_layers).to(cfg.device)
        self.supervisor = Supervisor(h, max(cfg.num_layers - 1, 1)).to(cfg.device)
        self.discriminator = Discriminator(h, cfg.num_layers).to(cfg.device)

        self.opt_ae = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=cfg.lr,
        )
        self.opt_sup = torch.optim.Adam(self.supervisor.parameters(), lr=cfg.lr)
        self.opt_g = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=cfg.lr,
        )
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.lr)

        self._mn = None
        self._mx = None

    def _to_loader(self, X: np.ndarray) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X))
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

    def fit(self, X_real: np.ndarray, verbose: bool = True) -> dict:
        cfg = self.cfg
        loader = self._to_loader(X_real)
        history = {"ae_loss": [], "sup_loss": [], "g_loss": [], "d_loss": []}

        # Phase 1: автокодировщик
        for ep in range(cfg.ae_epochs):
            losses = []
            for (xb,) in loader:
                xb = xb.to(cfg.device)
                h = self.embedder(xb)
                x_tilde = self.recovery(h)
                loss = F.mse_loss(x_tilde, xb)
                self.opt_ae.zero_grad()
                loss.backward()
                self.opt_ae.step()
                losses.append(loss.item())
            history["ae_loss"].append(float(np.mean(losses)))
            if verbose and (ep + 1) % max(1, cfg.ae_epochs // 5) == 0:
                print(f"[AE] {ep+1}/{cfg.ae_epochs}  loss={history['ae_loss'][-1]:.5f}", flush=True)

        # Phase 2: supervisor
        for ep in range(cfg.sup_epochs):
            losses = []
            for (xb,) in loader:
                xb = xb.to(cfg.device)
                with torch.no_grad():
                    h = self.embedder(xb)
                h_sup = self.supervisor(h)
                loss = F.mse_loss(h_sup[:, :-1, :], h[:, 1:, :])
                self.opt_sup.zero_grad()
                loss.backward()
                self.opt_sup.step()
                losses.append(loss.item())
            history["sup_loss"].append(float(np.mean(losses)))
            if verbose and (ep + 1) % max(1, cfg.sup_epochs // 5) == 0:
                print(f"[SUP] {ep+1}/{cfg.sup_epochs}  loss={history['sup_loss'][-1]:.5f}", flush=True)

        # Phase 3: совместная тренировка (BCE-GAN)
        for ep in range(cfg.joint_epochs):
            g_losses, d_losses = [], []
            for (xb,) in loader:
                xb = xb.to(cfg.device)
                B = xb.size(0)

                # ----- Generator step (G + S + E + R) -----
                z = torch.rand(B, cfg.seq_len, cfg.hidden_dim, device=cfg.device)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)

                d_fake_logits = self.discriminator(h_hat)
                g_adv = F.binary_cross_entropy_with_logits(
                    d_fake_logits, torch.ones_like(d_fake_logits)
                )

                h_real = self.embedder(xb)
                h_sup_real = self.supervisor(h_real)
                g_sup = F.mse_loss(h_sup_real[:, :-1, :], h_real[:, 1:, :])

                g_mean = torch.mean(torch.abs(xb.mean(dim=0) - x_hat.mean(dim=0)))
                g_std = torch.mean(
                    torch.abs(torch.sqrt(xb.var(dim=0) + 1e-6) - torch.sqrt(x_hat.var(dim=0) + 1e-6))
                )
                g_moments = g_mean + g_std

                x_tilde = self.recovery(h_real)
                g_ae = F.mse_loss(x_tilde, xb)

                g_loss = g_adv + 100.0 * g_sup + cfg.gamma_moment * g_moments + 10.0 * g_ae

                self.opt_g.zero_grad()
                self.opt_ae.zero_grad()
                g_loss.backward()
                self.opt_g.step()
                self.opt_ae.step()
                g_losses.append(g_loss.item())

                # ----- Discriminator step -----
                z = torch.rand(B, cfg.seq_len, cfg.hidden_dim, device=cfg.device)
                with torch.no_grad():
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    h_real_d = self.embedder(xb)
                d_real_logits = self.discriminator(h_real_d)
                d_fake_logits = self.discriminator(h_hat)
                d_loss = F.binary_cross_entropy_with_logits(
                    d_real_logits, torch.ones_like(d_real_logits)
                ) + F.binary_cross_entropy_with_logits(
                    d_fake_logits, torch.zeros_like(d_fake_logits)
                )
                self.opt_d.zero_grad()
                d_loss.backward()
                self.opt_d.step()
                d_losses.append(d_loss.item())

            history["g_loss"].append(float(np.mean(g_losses)))
            history["d_loss"].append(float(np.mean(d_losses)))
            if verbose and (ep + 1) % max(1, cfg.joint_epochs // 10) == 0:
                print(
                    f"[JOINT] {ep+1}/{cfg.joint_epochs} "
                    f"G={history['g_loss'][-1]:.4f}  D={history['d_loss'][-1]:.4f}",
                    flush=True,
                )

        return history

    @torch.no_grad()
    def sample_windows(self, n_windows: int) -> np.ndarray:
        cfg = self.cfg
        outs = []
        bs = max(1, min(n_windows, 256))
        for i in range(0, n_windows, bs):
            b = min(bs, n_windows - i)
            z = torch.rand(b, cfg.seq_len, cfg.hidden_dim, device=cfg.device)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat).cpu().numpy()
            outs.append(x_hat)
        return np.concatenate(outs, axis=0)


# ---------- удобный фасад ----------

def fit_and_sample(
    series_real: np.ndarray,
    n_target: int,
    cfg: TimeGANConfig | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict, "TimeGAN"]:
    cfg = cfg or TimeGANConfig()
    s = series_real.astype(np.float32)
    if s.ndim == 1:
        s = s[:, None]

    X = make_windows(s, T=cfg.seq_len, stride=1)
    mn, mx = fit_minmax(X)
    Xn = apply_minmax(X, mn, mx)

    model = TimeGAN(d=s.shape[1], cfg=cfg)
    model._mn = mn
    model._mx = mx
    history = model.fit(Xn, verbose=verbose)

    n_windows = int(np.ceil(n_target / cfg.seq_len)) + 1
    Xs_n = model.sample_windows(n_windows)
    Xs = invert_minmax(Xs_n, mn, mx)

    flat = Xs.reshape(-1, s.shape[1])[:n_target]
    return flat[:, 0], history, model
