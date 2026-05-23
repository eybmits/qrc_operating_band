#!/usr/bin/env python3
"""
Stateful dissipative gate-model QRC minimal-suite.

This script implements a density-matrix, state-carrying quantum reservoir:
    input Ry(beta * u_t) -> [frozen Rx mixer -> ZZ topology -> noise]^L
with linear ridge readout. It runs a compute-bounded version of the ablation
suite discussed in the accompanying conversation:
  - validation-defined operating regions + holdout test
  - gamma=0 and amplitude/dephasing/depolarizing noise controls
  - no-Rx/no-ZZ/Rx-only/ZZ-only/Rx-ZZ-Rx mixer controls
  - Z, ZZ, Z+ZZ, feature-count-matched readouts
  - uniform/masked/layer-reupload encodings
  - n/L/topology scaling probes
  - finite-shot and calibration perturbation probes
  - delay/NVAR/ESN classical baselines

The executed default is intentionally small enough to run on CPU. Increase GRID_MODE
or the split lengths for a paper-scale run.
"""
from __future__ import annotations

import os
# Important: many tiny BLAS calls are much faster without thread-spawning overhead.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.datasets import sunspots
except Exception:  # pragma: no cover
    sunspots = None

# -----------------------------
# Basic linear algebra
# -----------------------------
I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.diag([1, -1]).astype(np.complex128)


def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def rx(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * I2 - 1j * math.sin(theta / 2.0) * X2


def ry(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * I2 - 1j * math.sin(theta / 2.0) * Y2


def rz(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * I2 - 1j * math.sin(theta / 2.0) * Z2


def z_diagonals(n: int) -> np.ndarray:
    d = 2 ** n
    arr = []
    for i in range(n):
        vals = np.empty(d, dtype=float)
        shift = n - 1 - i
        for b in range(d):
            vals[b] = 1.0 if ((b >> shift) & 1) == 0 else -1.0
        arr.append(vals)
    return np.stack(arr, axis=0)


def edges_for_topology(n: int, topology: str) -> List[Tuple[int, int]]:
    if topology == "ring":
        return [(i, (i + 1) % n) for i in range(n)]
    if topology == "line":
        return [(i, i + 1) for i in range(n - 1)]
    if topology == "all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if topology == "skip_ring":
        return [(i, (i + 2) % n) for i in range(n)]
    raise ValueError(f"Unknown topology: {topology}")


def zz_phase_diag(n: int, edges: List[Tuple[int, int]], lam: float) -> np.ndarray:
    zds = z_diagonals(n)
    diag = np.zeros(2 ** n, dtype=float)
    for i, j in edges:
        diag += zds[i] * zds[j]
    return np.exp(-1j * lam * diag)


def full_noise_kraus(n: int, gamma: float, channel: str) -> List[np.ndarray]:
    """Full tensor-product Kraus list. Used for n>4 fallback."""
    d = 2 ** n
    if channel == "none" or gamma <= 0:
        return [np.eye(d, dtype=np.complex128)]
    if channel == "amplitude":
        local = [
            np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - gamma))]], dtype=np.complex128),
            np.array([[0.0, math.sqrt(max(0.0, gamma))], [0.0, 0.0]], dtype=np.complex128),
        ]
    elif channel == "dephasing":
        local = [math.sqrt(max(0.0, 1.0 - gamma)) * I2, math.sqrt(max(0.0, gamma)) * Z2]
    elif channel == "depolarizing":
        local = [
            math.sqrt(max(0.0, 1.0 - gamma)) * I2,
            math.sqrt(max(0.0, gamma / 3.0)) * X2,
            math.sqrt(max(0.0, gamma / 3.0)) * Y2,
            math.sqrt(max(0.0, gamma / 3.0)) * Z2,
        ]
    else:
        raise ValueError(f"Unknown channel: {channel}")
    ops: List[np.ndarray] = [np.array([[1.0 + 0j]])]
    for _ in range(n):
        ops = [np.kron(o, k) for o in ops for k in local]
    return ops


def noise_superop(n: int, gamma: float, channel: str) -> Optional[np.ndarray]:
    """Column-major superoperator. Only feasible/fast for n<=4."""
    d = 2 ** n
    if n > 4:
        return None
    if channel == "none" or gamma <= 0:
        return np.eye(d * d, dtype=np.complex128)
    kraus = full_noise_kraus(n, gamma, channel)
    s = np.zeros((d * d, d * d), dtype=np.complex128)
    for k in kraus:
        s += np.kron(k.conj(), k)
    return s


def apply_noise_density(rho: np.ndarray, kraus: List[np.ndarray]) -> np.ndarray:
    if len(kraus) == 1:
        k = kraus[0]
        if k.shape[0] == rho.shape[0] and np.allclose(k, np.eye(k.shape[0])):
            return rho
    out = np.zeros_like(rho)
    for k in kraus:
        out += k @ rho @ k.conj().T
    return out


def apply_local_noise_tensor(rho: np.ndarray, n: int, gamma: float, channel: str) -> np.ndarray:
    """Fast local product channels using 2x2 block formulas on each qubit.

    The density matrix is reshaped into row-bit axes followed by column-bit axes.
    For each qubit q, the pair of axes (q, n+q) carries the local 2x2 block.
    """
    if channel == "none" or gamma <= 0.0:
        return rho
    g = float(gamma)
    s = math.sqrt(max(0.0, 1.0 - g))
    d = rho.shape[0]
    t = rho.reshape((2,) * (2 * n))
    for q in range(n):
        tmp = np.moveaxis(t, [q, n + q], [0, 1])
        a = tmp[0, 0]
        b = tmp[0, 1]
        c = tmp[1, 0]
        dd = tmp[1, 1]
        new = np.empty_like(tmp)
        if channel == "amplitude":
            new[0, 0] = a + g * dd
            new[0, 1] = s * b
            new[1, 0] = s * c
            new[1, 1] = (1.0 - g) * dd
        elif channel == "dephasing":
            off = 1.0 - 2.0 * g
            new[0, 0] = a
            new[0, 1] = off * b
            new[1, 0] = off * c
            new[1, 1] = dd
        elif channel == "depolarizing":
            # For Kraus sqrt(1-g)I, sqrt(g/3){X,Y,Z}.
            new[0, 0] = (1.0 - 2.0 * g / 3.0) * a + (2.0 * g / 3.0) * dd
            new[1, 1] = (2.0 * g / 3.0) * a + (1.0 - 2.0 * g / 3.0) * dd
            off = 1.0 - 4.0 * g / 3.0
            new[0, 1] = off * b
            new[1, 0] = off * c
        else:
            raise ValueError(f"Unknown channel: {channel}")
        t = np.moveaxis(new, [0, 1], [q, n + q])
    return t.reshape((d, d))

# -----------------------------
# Tasks
# -----------------------------
@dataclass
class TaskData:
    name: str
    u: np.ndarray
    y: np.ndarray
    washout: int
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.washout + self.train + self.val + self.test


def _scale_to_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return 2.0 * (x - mn) / (mx - mn) - 1.0


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def make_mackey_glass(length: int, seed: int = 1, tau: int = 17) -> np.ndarray:
    # Discrete Euler form; good enough for a deterministic temporal benchmark.
    total = length + tau + 100
    x = np.empty(total, dtype=float)
    rng = np.random.default_rng(seed)
    x[: tau + 1] = 1.2 + 0.01 * rng.normal(size=tau + 1)
    a, b, c = 0.2, 0.1, 10
    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        x[t + 1] = x[t] + (a * x_tau / (1.0 + x_tau ** c) - b * x[t])
    return x[-length:]


def make_narma10(length: int, seed: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=length + 20)
    y = np.zeros(length + 20, dtype=float)
    for t in range(10, length + 19):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 9 : t + 1])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    return u[20:], y[20:]


def make_lorenz(length: int, dt: float = 0.02) -> np.ndarray:
    # RK4 integration, return x-coordinate after burn-in.
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    total = length + 1000
    state = np.array([1.0, 1.0, 1.0], dtype=float)
    xs = np.empty(total, dtype=float)

    def f(s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    for t in range(total):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[t] = state[0]
    return xs[-length:]


def make_sunspots_task() -> np.ndarray:
    if sunspots is None:
        # Deterministic fallback: noisy 11-year-like cycle.
        rng = np.random.default_rng(3)
        t = np.arange(320)
        return 50 + 40 * np.sin(2 * np.pi * t / 11.0) + 12 * rng.normal(size=len(t))
    data = sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(dtype=float)
    return data


def make_tasks(profile: str = "lite") -> Dict[str, TaskData]:
    if profile == "tiny":
        synth_splits = (50, 220, 70, 70)
    elif profile == "paperish":
        synth_splits = (300, 1600, 400, 400)
    else:
        synth_splits = (80, 400, 120, 120)
    w, tr, va, te = synth_splits
    total = w + tr + va + te + 1

    tasks: Dict[str, TaskData] = {}

    mg = make_mackey_glass(total + 1)
    u = _scale_to_unit(mg[:-1])
    y = _zscore(mg[1:])
    tasks["mackey_glass"] = TaskData("mackey_glass", u[: total], y[: total], w, tr, va, te)

    un, yn = make_narma10(total)
    tasks["narma10"] = TaskData("narma10", _scale_to_unit(un[:total]), _zscore(yn[:total]), w, tr, va, te)

    lo = make_lorenz(total + 1)
    tasks["lorenz"] = TaskData("lorenz", _scale_to_unit(lo[:-1]), _zscore(lo[1:]), w, tr, va, te)

    ss = make_sunspots_task()
    # annual series is short; keep a smaller chronological split.
    sw, strn, sva, ste = (20, 150, 50, 60)
    if len(ss) < sw + strn + sva + ste + 1:
        ste = max(30, len(ss) - sw - strn - sva - 1)
    ss_total = sw + strn + sva + ste
    tasks["sunspots_annual"] = TaskData(
        "sunspots_annual", _scale_to_unit(ss[:-1])[:ss_total], _zscore(ss[1:])[:ss_total], sw, strn, sva, ste
    )
    return tasks

# -----------------------------
# Reservoir config and simulator
# -----------------------------
@dataclass(frozen=True)
class QRCConfig:
    n: int = 4
    layers: int = 2
    beta: float = 0.1 * math.pi
    lam: float = 0.25 * math.pi
    gamma: float = 0.1
    channel: str = "amplitude"  # amplitude, dephasing, depolarizing, none
    topology: str = "ring"
    mixer: str = "rx_zz"  # rx_zz, rx_only, zz_only, none, rx_zz_rx
    input_mode: str = "uniform"  # uniform, masked, reupload
    seed: int = 42


def build_input_unitaries(task: TaskData, cfg: QRCConfig, mask: np.ndarray) -> List[np.ndarray]:
    mats = []
    n = cfg.n
    for u in task.u[: task.total]:
        if cfg.input_mode == "masked":
            angles = cfg.beta * float(u) * mask
        else:
            angles = np.full(n, cfg.beta * float(u), dtype=float)
        mats.append(kron_all([ry(float(a)) for a in angles]))
    return mats


def build_layer_data(cfg: QRCConfig) -> Dict[str, Any]:
    n = cfg.n
    d = 2 ** n
    rng = np.random.default_rng(cfg.seed)
    theta1 = rng.uniform(0.0, 2 * np.pi, size=n)
    theta2 = rng.uniform(0.0, 2 * np.pi, size=n)
    U_rx1 = kron_all([rx(float(t)) for t in theta1])
    U_rx2 = kron_all([rx(float(t)) for t in theta2])
    edges = edges_for_topology(n, cfg.topology)
    zz_diag = zz_phase_diag(n, edges, cfg.lam)
    # Unitarios by mixer mode.
    if cfg.mixer == "rx_zz":
        U_layer = zz_diag[:, None] * U_rx1
        U_post = None
    elif cfg.mixer == "rx_only":
        U_layer = U_rx1
        U_post = None
    elif cfg.mixer == "zz_only":
        U_layer = np.diag(zz_diag)
        U_post = None
    elif cfg.mixer == "none":
        U_layer = np.eye(d, dtype=np.complex128)
        U_post = None
    elif cfg.mixer == "rx_zz_rx":
        U_layer = zz_diag[:, None] * U_rx1
        U_post = U_rx2
    else:
        raise ValueError(f"Unknown mixer: {cfg.mixer}")
    if U_post is not None:
        U_layer = U_post @ U_layer

    zds = z_diagonals(n)
    zzds = np.stack([zds[i] * zds[j] for i, j in edges], axis=0) if edges else np.zeros((0, d))

    # Superoperator for n<=4; density fallback for n>4.
    super_noise = noise_superop(n, cfg.gamma, cfg.channel)
    if super_noise is not None:
        S_layer = super_noise @ np.kron(U_layer.conj(), U_layer)
        kraus = None
    else:
        S_layer = None
        kraus = None
    return {"edges": edges, "zds": zds, "zzds": zzds, "S_layer": S_layer, "U_layer": U_layer, "kraus": kraus}


def simulate_features(task: TaskData, cfg: QRCConfig, uin_cache: Optional[Dict[str, List[np.ndarray]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return Xz and Xzz after washout for all train+val+test time steps."""
    n, d = cfg.n, 2 ** cfg.n
    rng = np.random.default_rng(cfg.seed + 1009)
    if cfg.input_mode == "masked":
        mask = rng.choice([-1.0, 1.0], size=n)
        # Avoid an all-symmetry case; add mild fixed heterogeneity.
        mask = mask * rng.uniform(0.5, 1.5, size=n)
    else:
        mask = np.ones(n, dtype=float)

    key = f"{task.name}|n={n}|beta={cfg.beta:.12g}|input={cfg.input_mode}|seed={cfg.seed}"
    if uin_cache is not None and key in uin_cache:
        Uins = uin_cache[key]
    else:
        Uins = build_input_unitaries(task, cfg, mask)
        if uin_cache is not None:
            uin_cache[key] = Uins

    layer = build_layer_data(cfg)
    zds = layer["zds"]
    zzds = layer["zzds"]
    S_layer = layer["S_layer"]
    U_layer = layer["U_layer"]
    kraus = layer["kraus"]

    rho = np.zeros((d, d), dtype=np.complex128)
    rho[0, 0] = 1.0
    Xz: List[np.ndarray] = []
    Xzz: List[np.ndarray] = []
    for t in range(task.total):
        Uin = Uins[t]
        if cfg.input_mode != "reupload":
            rho = Uin @ rho @ Uin.conj().T
        z_feats = []
        zz_feats = []
        for _ in range(cfg.layers):
            if cfg.input_mode == "reupload":
                rho = Uin @ rho @ Uin.conj().T
            if S_layer is not None:
                v = rho.reshape(-1, order="F")
                v = S_layer @ v
                rho = v.reshape((d, d), order="F")
            else:
                rho = U_layer @ rho @ U_layer.conj().T
                rho = apply_local_noise_tensor(rho, cfg.n, cfg.gamma, cfg.channel)
            diag = np.real(np.diag(rho))
            z_feats.append(zds @ diag)
            if zzds.shape[0] > 0:
                zz_feats.append(zzds @ diag)
            else:
                zz_feats.append(np.zeros(0, dtype=float))
        if t >= task.washout:
            Xz.append(np.concatenate(z_feats))
            Xzz.append(np.concatenate(zz_feats))
    return np.asarray(Xz, dtype=float), np.asarray(Xzz, dtype=float)

# -----------------------------
# Readout/evaluation
# -----------------------------
def split_indices(task: TaskData) -> Tuple[slice, slice, slice]:
    tr = slice(0, task.train)
    va = slice(task.train, task.train + task.val)
    te = slice(task.train + task.val, task.train + task.val + task.test)
    return tr, va, te


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones(len(X))])


def fit_ridge_closed(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xa = add_intercept(X)
    A = Xa.T @ Xa
    reg = np.eye(A.shape[0]) * alpha
    reg[-1, -1] = 0.0  # don't penalize intercept
    b = Xa.T @ y
    try:
        return np.linalg.solve(A + reg, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg) @ b


def pred_ridge(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ w


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2) / (np.var(y_true) + 1e-12))


def evaluate_readout(X: np.ndarray, y_all: np.ndarray, task: TaskData, alphas: np.ndarray) -> Dict[str, float]:
    tr, va, te = split_indices(task)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y_all[task.washout : task.total][tr], y_all[task.washout : task.total][va], y_all[task.washout : task.total][te]
    best = None
    for a in alphas:
        w = fit_ridge_closed(Xtr, ytr, float(a))
        score = nmse(yva, pred_ridge(Xva, w))
        if best is None or score < best[0]:
            best = (score, float(a), w)
    assert best is not None
    val_nmse, alpha, w = best
    return {"val_nmse": float(val_nmse), "test_nmse": nmse(yte, pred_ridge(Xte, w)), "alpha": alpha}


def make_readout_matrices(Xz: np.ndarray, Xzz: np.ndarray, seed: int = 0) -> Dict[str, np.ndarray]:
    out = {
        "Z": Xz,
        "ZZ": Xzz if Xzz.shape[1] > 0 else np.zeros((len(Xz), 1)),
        "Z+ZZ": np.column_stack([Xz, Xzz]) if Xzz.shape[1] > 0 else Xz,
    }
    full = out["Z+ZZ"]
    k = Xz.shape[1]
    if full.shape[1] <= k:
        out["matched"] = full
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(full.shape[1], size=k, replace=False))
        out["matched"] = full[:, idx]
    return out

# -----------------------------
# Classical baselines
# -----------------------------
def build_delay_features(u: np.ndarray, max_delay: int = 20, delays: Optional[List[int]] = None) -> Tuple[np.ndarray, int]:
    if delays is None:
        delays = list(range(max_delay + 1))
    start = max(delays)
    X = []
    for t in range(start, len(u)):
        X.append([u[t - d] for d in delays])
    return np.asarray(X, dtype=float), start


def build_nvar_features(u: np.ndarray, delays: Optional[List[int]] = None) -> Tuple[np.ndarray, int]:
    if delays is None:
        delays = [0, 1, 2, 4, 8, 16, 24]
    Xlin, start = build_delay_features(u, delays=delays)
    # quadratic terms, upper triangular
    cols = [Xlin]
    quad = []
    for i in range(Xlin.shape[1]):
        for j in range(i, Xlin.shape[1]):
            quad.append(Xlin[:, i] * Xlin[:, j])
    cols.append(np.stack(quad, axis=1))
    return np.column_stack(cols), start


def evaluate_static_features(X_full: np.ndarray, y: np.ndarray, start: int, task: TaskData, alphas: np.ndarray) -> Dict[str, float]:
    # Align to task splits: QRC uses y indices [washout:total]. For delay baselines, use same y period plus delay start.
    base_start = task.washout
    idx0 = max(base_start, start)
    idx1 = task.total
    X = X_full[idx0 - start : idx1 - start]
    yy = y[idx0:idx1]
    # Adjust split lengths if delay eats into washout only; usually no effect.
    lost = idx0 - base_start
    tr_len = max(10, task.train - lost)
    va_len = task.val
    te_len = min(task.test, len(yy) - tr_len - va_len)
    tr = slice(0, tr_len)
    va = slice(tr_len, tr_len + va_len)
    te = slice(tr_len + va_len, tr_len + va_len + te_len)
    best = None
    for a in alphas:
        w = fit_ridge_closed(X[tr], yy[tr], float(a))
        score = nmse(yy[va], pred_ridge(X[va], w))
        if best is None or score < best[0]:
            best = (score, float(a), w)
    assert best is not None
    return {"val_nmse": float(best[0]), "test_nmse": nmse(yy[te], pred_ridge(X[te], best[2])), "alpha": best[1]}


def run_esn(task: TaskData, units: int = 100, seed: int = 0, alphas: Optional[np.ndarray] = None) -> Dict[str, float]:
    if alphas is None:
        alphas = np.logspace(-8, 3, 12)
    rng = np.random.default_rng(seed)
    # Hyperparameter mini-search on validation.
    candidates = [(0.6, 0.5, 0.3), (0.9, 0.5, 0.3), (0.9, 1.0, 0.5), (1.1, 0.2, 0.2)]
    y_period = task.y[task.washout : task.total]
    tr, va, te = split_indices(task)
    best_global = None
    for sr, ins, leak in candidates:
        W = rng.normal(size=(units, units)) / math.sqrt(units)
        # Spectral radius scaling.
        eig = np.linalg.eigvals(W)
        W *= sr / (np.max(np.abs(eig)) + 1e-12)
        Win = rng.normal(size=(units, 1)) * ins
        b = rng.normal(size=units) * 0.01
        x = np.zeros(units)
        Xstates = []
        for t in range(task.total):
            pre = W @ x + Win[:, 0] * task.u[t] + b
            x = (1.0 - leak) * x + leak * np.tanh(pre)
            if t >= task.washout:
                Xstates.append(x.copy())
        Xstates = np.asarray(Xstates)
        for a in alphas:
            w = fit_ridge_closed(Xstates[tr], y_period[tr], float(a))
            v = nmse(y_period[va], pred_ridge(Xstates[va], w))
            if best_global is None or v < best_global[0]:
                best_global = (v, float(a), w, Xstates, sr, ins, leak)
    assert best_global is not None
    _, a, w, Xstates, sr, ins, leak = best_global
    return {
        "val_nmse": float(best_global[0]),
        "test_nmse": nmse(y_period[te], pred_ridge(Xstates[te], w)),
        "alpha": a,
        "spectral_radius": sr,
        "input_scale": ins,
        "leak": leak,
    }

# -----------------------------
# Suite runners
# -----------------------------
def grid_values(mode: str = "lite") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "tiny":
        beta_fracs = np.array([0.0, 0.06, 0.15, 0.5])
        lam_fracs = np.array([0.0, 0.18, 0.35, 0.5])
        gammas = np.array([0.0, 0.07, 0.18, 0.3])
    elif mode == "paperish":
        beta_fracs = np.linspace(0.0, 0.5, 9)
        lam_fracs = np.linspace(0.0, 0.5, 9)
        gammas = np.linspace(0.0, 0.3, 13)
    else:
        beta_fracs = np.array([0.0, 0.04, 0.10, 0.22, 0.5])
        lam_fracs = np.array([0.0, 0.12, 0.25, 0.38, 0.5])
        gammas = np.array([0.0, 0.05, 0.12, 0.22, 0.3])
    return beta_fracs * np.pi, lam_fracs * np.pi, gammas


def run_main_grid(tasks: Dict[str, TaskData], outdir: Path, grid_mode: str = "lite", seed: int = 42) -> pd.DataFrame:
    alphas = np.logspace(-8, 3, 12)
    betas, lams, gammas = grid_values(grid_mode)
    rows = []
    uin_cache: Dict[str, List[np.ndarray]] = {}
    t0 = time.time()
    total = len(betas) * len(lams) * len(gammas) * len(tasks)
    done = 0
    for beta in betas:
        for lam in lams:
            for gamma in gammas:
                cfg = QRCConfig(n=4, layers=2, beta=float(beta), lam=float(lam), gamma=float(gamma), channel="amplitude", topology="ring", mixer="rx_zz", input_mode="uniform", seed=seed)
                for task in tasks.values():
                    Xz, Xzz = simulate_features(task, cfg, uin_cache)
                    mats = make_readout_matrices(Xz, Xzz, seed=seed)
                    for readout, Xmat in mats.items():
                        res = evaluate_readout(Xmat, task.y, task, alphas)
                        rows.append({
                            "suite": "main_grid",
                            "task": task.name,
                            "readout": readout,
                            "beta": cfg.beta,
                            "beta_pi": cfg.beta / np.pi,
                            "lambda": cfg.lam,
                            "lambda_pi": cfg.lam / np.pi,
                            "gamma": cfg.gamma,
                            "channel": cfg.channel,
                            "mixer": cfg.mixer,
                            "input_mode": cfg.input_mode,
                            "topology": cfg.topology,
                            "n": cfg.n,
                            "layers": cfg.layers,
                            "seed": cfg.seed,
                            **res,
                        })
                    done += 1
                    if done % 50 == 0:
                        print(f"main grid {done}/{total} sims, elapsed {time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "qrc_main_grid_results.csv", index=False)
    return df


def validation_regions(df: pd.DataFrame, readout: str = "Z+ZZ") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # one row per grid point per task for selected readout
    d = df[df["readout"] == readout].copy()
    key_cols = ["beta_pi", "lambda_pi", "gamma"]
    # rank per task by validation NMSE
    d["val_rank_pct"] = d.groupby("task")["val_nmse"].rank(method="average", pct=True)
    d["test_rank_pct"] = d.groupby("task")["test_nmse"].rank(method="average", pct=True)
    # Compute intersections based on validation top p.
    summary = {}
    n_points = d[key_cols].drop_duplicates().shape[0]
    for p in [5, 10, 20]:
        sets = []
        for task, g in d.groupby("task"):
            thr = np.percentile(g["val_nmse"], p)
            pts = set(tuple(x) for x in g[g["val_nmse"] <= thr][key_cols].to_numpy())
            sets.append(pts)
        inter = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        summary[f"top{p}_observed"] = len(inter)
        summary[f"top{p}_union"] = len(union)
        summary[f"top{p}_null"] = n_points * (p / 100.0) ** len(sets)
        if inter:
            mask = d[key_cols].apply(lambda r: tuple(r.to_numpy()) in inter, axis=1)
            summary[f"top{p}_region_test_mean"] = float(d[mask]["test_nmse"].mean())
            summary[f"top{p}_region_test_median"] = float(d[mask]["test_nmse"].median())
            arr = np.array(list(inter), dtype=float)
            med = np.median(arr, axis=0)
            # choose closest actual grid point to the coordinate-wise median
            idx = np.argmin(np.sum((arr - med) ** 2, axis=1))
            summary[f"top{p}_rep_beta_pi"] = float(arr[idx, 0])
            summary[f"top{p}_rep_lambda_pi"] = float(arr[idx, 1])
            summary[f"top{p}_rep_gamma"] = float(arr[idx, 2])
        else:
            summary[f"top{p}_region_test_mean"] = np.nan
            summary[f"top{p}_region_test_median"] = np.nan
            summary[f"top{p}_rep_beta_pi"] = np.nan
            summary[f"top{p}_rep_lambda_pi"] = np.nan
            summary[f"top{p}_rep_gamma"] = np.nan
    # If top intersections are empty, representative is best mean validation rank.
    grouped = d.groupby(key_cols).agg(mean_val_rank=("val_rank_pct", "mean"), mean_test_nmse=("test_nmse", "mean"), mean_val_nmse=("val_nmse", "mean")).reset_index()
    best = grouped.sort_values("mean_val_rank").iloc[0].to_dict()
    summary["rank_rep_beta_pi"] = float(best["beta_pi"])
    summary["rank_rep_lambda_pi"] = float(best["lambda_pi"])
    summary["rank_rep_gamma"] = float(best["gamma"])
    summary["rank_rep_mean_test_nmse"] = float(best["mean_test_nmse"])
    return d, summary


def pick_representative(summary: Dict[str, Any]) -> Tuple[float, float, float, str]:
    for p in [5, 10, 20]:
        b = summary.get(f"top{p}_rep_beta_pi")
        if b is not None and not (isinstance(b, float) and np.isnan(b)):
            return float(b) * np.pi, float(summary[f"top{p}_rep_lambda_pi"]) * np.pi, float(summary[f"top{p}_rep_gamma"]), f"top{p}_intersection"
    return float(summary["rank_rep_beta_pi"]) * np.pi, float(summary["rank_rep_lambda_pi"]) * np.pi, float(summary["rank_rep_gamma"]), "mean_validation_rank"


def evaluate_cfg_on_tasks(cfg: QRCConfig, tasks: Dict[str, TaskData], readout: str = "Z+ZZ", shots: Optional[int] = None, shot_reps: int = 1, perturb_label: str = "") -> List[Dict[str, Any]]:
    alphas = np.logspace(-8, 3, 12)
    rows = []
    uin_cache: Dict[str, List[np.ndarray]] = {}
    rng = np.random.default_rng(cfg.seed + 777)
    for task in tasks.values():
        Xz, Xzz = simulate_features(task, cfg, uin_cache)
        mats = make_readout_matrices(Xz, Xzz, seed=cfg.seed)
        Xbase = mats[readout]
        reps = shot_reps if shots is not None else 1
        for r in range(reps):
            X = Xbase.copy()
            if shots is not None:
                # Gaussian approximation to measurement noise for +/-1 observables.
                # For features outside [-1,1] numerical clipping keeps variance sane.
                mu = np.clip(X, -1.0, 1.0)
                sigma = np.sqrt(np.maximum(0.0, 1.0 - mu ** 2) / float(shots))
                X = mu + rng.normal(scale=sigma, size=mu.shape)
            res = evaluate_readout(X, task.y, task, alphas)
            rows.append({
                "task": task.name,
                "readout": readout,
                "shots": shots if shots is not None else "exact",
                "shot_rep": r,
                "perturb_label": perturb_label,
                "beta": cfg.beta,
                "beta_pi": cfg.beta / np.pi,
                "lambda": cfg.lam,
                "lambda_pi": cfg.lam / np.pi,
                "gamma": cfg.gamma,
                "channel": cfg.channel,
                "mixer": cfg.mixer,
                "input_mode": cfg.input_mode,
                "topology": cfg.topology,
                "n": cfg.n,
                "layers": cfg.layers,
                "seed": cfg.seed,
                **res,
            })
    return rows


def run_focused_ablations(tasks: Dict[str, TaskData], outdir: Path, rep: Tuple[float, float, float, str], seed: int = 42) -> pd.DataFrame:
    beta, lam, gamma, rep_source = rep
    rows = []
    def add(label: str, cfg: QRCConfig):
        print(f"ablation: {label}", flush=True)
        for row in evaluate_cfg_on_tasks(cfg, tasks, readout="Z+ZZ"):
            row.update({"suite": "focused_ablation", "ablation": label, "rep_source": rep_source})
            rows.append(row)

    base = dict(n=4, layers=2, beta=beta, lam=lam, gamma=gamma, channel="amplitude", topology="ring", mixer="rx_zz", input_mode="uniform", seed=seed)
    add("base_exact_AD", QRCConfig(**base))

    # Noise controls.
    add("gamma0_AD", QRCConfig(**{**base, "gamma": 0.0}))
    add("dephasing", QRCConfig(**{**base, "channel": "dephasing"}))
    add("depolarizing", QRCConfig(**{**base, "channel": "depolarizing"}))
    add("no_noise_channel", QRCConfig(**{**base, "channel": "none", "gamma": 0.0}))

    # Mixer controls.
    for mix in ["none", "zz_only", "rx_only", "rx_zz_rx"]:
        add(f"mixer_{mix}", QRCConfig(**{**base, "mixer": mix}))

    # Input encoding controls.
    add("input_masked", QRCConfig(**{**base, "input_mode": "masked"}))
    add("input_reupload", QRCConfig(**{**base, "input_mode": "reupload"}))

    # n/L/topology scaling probes.
    add("layers_1", QRCConfig(**{**base, "layers": 1}))
    add("layers_4", QRCConfig(**{**base, "layers": 4}))
    add("topology_line", QRCConfig(**{**base, "topology": "line"}))
    add("topology_all", QRCConfig(**{**base, "topology": "all"}))
    # n=6 exact density fallback; keep topology ring.
    add("n6_ring", QRCConfig(**{**base, "n": 6}))

    # Finite-shot probes on the representative point.
    for shots in [128, 512, 2048]:
        print(f"finite shots: {shots}", flush=True)
        for row in evaluate_cfg_on_tasks(QRCConfig(**base), tasks, readout="Z+ZZ", shots=shots, shot_reps=3):
            row.update({"suite": "finite_shot", "ablation": f"shots_{shots}", "rep_source": rep_source})
            rows.append(row)

    # Calibration perturbation around the region representative.
    rng = np.random.default_rng(seed + 999)
    for k in range(8):
        b2 = float(np.clip(beta / np.pi + rng.normal(0, 0.015), 0.0, 0.5) * np.pi)
        l2 = float(np.clip(lam / np.pi + rng.normal(0, 0.025), 0.0, 0.5) * np.pi)
        g2 = float(np.clip(gamma + rng.normal(0, 0.02), 0.0, 0.3))
        cfg = QRCConfig(**{**base, "beta": b2, "lam": l2, "gamma": g2})
        for row in evaluate_cfg_on_tasks(cfg, tasks, readout="Z+ZZ", perturb_label=f"calib_{k}"):
            row.update({"suite": "calibration", "ablation": "calibration_perturb", "rep_source": rep_source})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "qrc_focused_ablation_results.csv", index=False)
    return df


def run_classical_baselines(tasks: Dict[str, TaskData], outdir: Path) -> pd.DataFrame:
    alphas = np.logspace(-8, 3, 12)
    rows = []
    for task in tasks.values():
        print(f"classical baselines: {task.name}", flush=True)
        Xd, start = build_delay_features(task.u[: task.total], max_delay=20)
        res = evaluate_static_features(Xd, task.y, start, task, alphas)
        rows.append({"baseline": "delay_ridge", "task": task.name, **res})
        Xn, start = build_nvar_features(task.u[: task.total])
        res = evaluate_static_features(Xn, task.y, start, task, alphas)
        rows.append({"baseline": "NVAR_degree2", "task": task.name, **res})
        res = run_esn(task, units=100, seed=123, alphas=alphas)
        rows.append({"baseline": "ESN_100", "task": task.name, **res})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "qrc_classical_baselines.csv", index=False)
    return df

# -----------------------------
# Plotting and summarization
# -----------------------------
def percentile_rank(x: pd.Series) -> pd.Series:
    return x.rank(method="average", pct=True)


def make_phase_plots(region_df: pd.DataFrame, outdir: Path, readout: str = "Z+ZZ") -> None:
    d = region_df.copy()
    # mean validation percentile rank across tasks at each grid point
    grouped = d.groupby(["beta_pi", "lambda_pi", "gamma"]).agg(mean_rank=("val_rank_pct", "mean"), mean_test=("test_nmse", "mean")).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    projections = [
        ("beta_pi", "gamma", "lambda_pi", r"$\beta/\pi$", r"$\gamma$", r"min over $\lambda$"),
        ("lambda_pi", "gamma", "beta_pi", r"$\lambda/\pi$", r"$\gamma$", r"min over $\beta$"),
        ("lambda_pi", "beta_pi", "gamma", r"$\lambda/\pi$", r"$\beta/\pi$", r"min over $\gamma$"),
    ]
    for ax, (xcol, ycol, omit, xlabel, ylabel, title) in zip(axes, projections):
        proj = grouped.groupby([xcol, ycol])["mean_rank"].min().reset_index()
        xs = np.sort(proj[xcol].unique())
        ys = np.sort(proj[ycol].unique())
        Z = np.full((len(ys), len(xs)), np.nan)
        for _, r in proj.iterrows():
            ix = np.where(xs == r[xcol])[0][0]
            iy = np.where(ys == r[ycol])[0][0]
            Z[iy, ix] = r["mean_rank"]
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=[xs[0], xs[-1], ys[0], ys[-1]])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # outline low-rank region if possible
        try:
            ax.contour(xs, ys, Z, levels=[np.nanpercentile(Z, 20)], linewidths=1.5)
        except Exception:
            pass
    fig.colorbar(im, ax=axes, label="cross-task validation rank, lower is better")
    fig.suptitle(f"Stateful QRC validation-defined operating-region projections ({readout})")
    fig.savefig(outdir / "qrc_phase_maps_validation.png", dpi=180)
    plt.close(fig)


def make_summary_plots(main_df: pd.DataFrame, region_summary: Dict[str, Any], abl_df: pd.DataFrame, base_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # Cross-task overlap based on validation regions.
    ps = [5, 10, 20]
    obs = [region_summary.get(f"top{p}_observed", np.nan) for p in ps]
    null = [region_summary.get(f"top{p}_null", np.nan) for p in ps]
    x = np.arange(len(ps))
    width = 0.35
    axes[0, 0].bar(x - width / 2, obs, width, label="observed")
    axes[0, 0].bar(x + width / 2, null, width, label="independent null")
    axes[0, 0].set_yscale("symlog", linthresh=0.1)
    axes[0, 0].set_xticks(x, [f"top {p}%" for p in ps])
    axes[0, 0].set_title("validation top-set overlap")
    axes[0, 0].set_ylabel("shared grid points")
    axes[0, 0].legend()

    # Readout ablations: best mean test by readout.
    read = main_df.groupby(["readout", "beta_pi", "lambda_pi", "gamma"]).agg(mean_test=("test_nmse", "mean"), mean_val=("val_nmse", "mean")).reset_index()
    best_read = read.sort_values("mean_val").groupby("readout").head(1).sort_values("mean_test")
    axes[0, 1].bar(best_read["readout"], best_read["mean_test"])
    axes[0, 1].set_title("readout ablations, val-selected")
    axes[0, 1].set_ylabel("mean holdout NMSE")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Noise ablations.
    tmp = abl_df[abl_df["ablation"].isin(["base_exact_AD", "gamma0_AD", "dephasing", "depolarizing", "no_noise_channel"])].groupby("ablation")["test_nmse"].mean().reset_index()
    axes[0, 2].bar(tmp["ablation"], tmp["test_nmse"])
    axes[0, 2].set_title("noise channel controls")
    axes[0, 2].set_ylabel("mean holdout NMSE")
    axes[0, 2].tick_params(axis="x", rotation=45)

    # Mixer/input/topology/layers.
    sel = ["base_exact_AD", "mixer_none", "mixer_zz_only", "mixer_rx_only", "mixer_rx_zz_rx", "input_masked", "input_reupload"]
    tmp = abl_df[abl_df["ablation"].isin(sel)].groupby("ablation")["test_nmse"].mean().reindex(sel).reset_index()
    axes[1, 0].bar(tmp["ablation"], tmp["test_nmse"])
    axes[1, 0].set_title("mixer + input controls")
    axes[1, 0].set_ylabel("mean holdout NMSE")
    axes[1, 0].tick_params(axis="x", rotation=55)

    sel = ["base_exact_AD", "layers_1", "layers_4", "topology_line", "topology_all", "n6_ring"]
    tmp = abl_df[abl_df["ablation"].isin(sel)].groupby("ablation")["test_nmse"].mean().reindex(sel).reset_index()
    axes[1, 1].bar(tmp["ablation"], tmp["test_nmse"])
    axes[1, 1].set_title("n/L/topology scaling probes")
    axes[1, 1].set_ylabel("mean holdout NMSE")
    axes[1, 1].tick_params(axis="x", rotation=45)

    # Classical baselines vs QRC representative.
    qrc_mean = abl_df[abl_df["ablation"] == "base_exact_AD"].groupby("task")["test_nmse"].mean().reset_index()
    qrc_mean["method"] = "QRC_rep"
    base_plot = base_df.rename(columns={"baseline": "method"})[["method", "task", "test_nmse"]]
    comp = pd.concat([qrc_mean[["method", "task", "test_nmse"]], base_plot], ignore_index=True)
    comp_mean = comp.groupby("method")["test_nmse"].mean().sort_values().reset_index()
    axes[1, 2].bar(comp_mean["method"], comp_mean["test_nmse"])
    axes[1, 2].set_title("classical baselines vs QRC representative")
    axes[1, 2].set_ylabel("mean holdout NMSE")
    axes[1, 2].tick_params(axis="x", rotation=45)

    fig.suptitle("Stateful QRC minimal-suite summary")
    fig.savefig(outdir / "qrc_minimal_suite_summary.png", dpi=180)
    plt.close(fig)

    # Robustness plot.
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    shot = abl_df[abl_df["suite"].isin(["finite_shot", "focused_ablation"])]
    rows = []
    rows.append(("exact", float(abl_df[abl_df["ablation"] == "base_exact_AD"]["test_nmse"].mean())))
    for shots in [128, 512, 2048]:
        rows.append((f"{shots} shots", float(abl_df[abl_df["ablation"] == f"shots_{shots}"]["test_nmse"].mean())))
    rows.append(("calib pert.", float(abl_df[abl_df["ablation"] == "calibration_perturb"]["test_nmse"].mean())))
    lab, val = zip(*rows)
    ax.bar(lab, val)
    ax.set_title("finite-shot approximation and calibration perturbation")
    ax.set_ylabel("mean holdout NMSE")
    fig.savefig(outdir / "qrc_robustness_summary.png", dpi=180)
    plt.close(fig)


def summarize_tables(main_df: pd.DataFrame, region_df: pd.DataFrame, region_summary: Dict[str, Any], abl_df: pd.DataFrame, base_df: pd.DataFrame, outdir: Path) -> Dict[str, Any]:
    # Main readout summary.
    read = main_df.groupby(["readout", "beta_pi", "lambda_pi", "gamma"]).agg(mean_test=("test_nmse", "mean"), mean_val=("val_nmse", "mean")).reset_index()
    read_best = read.sort_values("mean_val").groupby("readout").head(1).sort_values("mean_test")

    abl_summary = abl_df.groupby("ablation").agg(mean_test_nmse=("test_nmse", "mean"), median_test_nmse=("test_nmse", "median"), n_rows=("test_nmse", "size")).reset_index().sort_values("mean_test_nmse")
    base_summary = base_df.groupby("baseline").agg(mean_test_nmse=("test_nmse", "mean"), median_test_nmse=("test_nmse", "median")).reset_index().sort_values("mean_test_nmse")

    read_best.to_csv(outdir / "qrc_readout_best_summary.csv", index=False)
    abl_summary.to_csv(outdir / "qrc_ablation_summary.csv", index=False)
    base_summary.to_csv(outdir / "qrc_baseline_summary.csv", index=False)

    summary = {
        "region_summary": region_summary,
        "readout_best": read_best.to_dict(orient="records"),
        "ablation_summary": abl_summary.to_dict(orient="records"),
        "baseline_summary": base_summary.to_dict(orient="records"),
    }
    with open(outdir / "qrc_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # human-readable markdown
    with open(outdir / "qrc_run_report.md", "w") as f:
        f.write("# Stateful QRC minimal-suite report\n\n")
        f.write("## Region summary\n\n")
        for k, v in region_summary.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Val-selected readout bests\n\n")
        f.write(read_best.to_markdown(index=False))
        f.write("\n\n## Focused ablations\n\n")
        f.write(abl_summary.to_markdown(index=False))
        f.write("\n\n## Classical baselines\n\n")
        f.write(base_summary.to_markdown(index=False))
        f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/qrc_stateful_run")
    parser.add_argument("--grid-mode", type=str, default="lite", choices=["tiny", "lite", "paperish"])
    parser.add_argument("--task-profile", type=str, default="lite", choices=["tiny", "lite", "paperish"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks = make_tasks(args.task_profile)
    task_meta = {name: {"washout": t.washout, "train": t.train, "val": t.val, "test": t.test, "total": t.total} for name, t in tasks.items()}
    with open(outdir / "qrc_task_metadata.json", "w") as f:
        json.dump(task_meta, f, indent=2)
    print("tasks:", task_meta, flush=True)

    main_df = run_main_grid(tasks, outdir, grid_mode=args.grid_mode, seed=args.seed)
    region_df, region_summary = validation_regions(main_df, readout="Z+ZZ")
    with open(outdir / "qrc_region_summary.json", "w") as f:
        json.dump(region_summary, f, indent=2)
    make_phase_plots(region_df, outdir, readout="Z+ZZ")
    rep = pick_representative(region_summary)
    print("representative:", rep, flush=True)

    if args.skip_ablations:
        abl_df = pd.DataFrame()
        base_df = pd.DataFrame()
    else:
        abl_df = run_focused_ablations(tasks, outdir, rep, seed=args.seed)
        base_df = run_classical_baselines(tasks, outdir)
        make_summary_plots(main_df, region_summary, abl_df, base_df, outdir)
        summarize_tables(main_df, region_df, region_summary, abl_df, base_df, outdir)

    # record command and elapsed time
    with open(outdir / "qrc_run_info.json", "w") as f:
        json.dump({"grid_mode": args.grid_mode, "task_profile": args.task_profile, "seed": args.seed, "elapsed_seconds": time.time() - t0}, f, indent=2)
    print(f"done in {time.time() - t0:.1f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
