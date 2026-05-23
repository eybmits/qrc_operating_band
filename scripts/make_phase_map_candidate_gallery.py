#!/usr/bin/env python3
"""Generate optional phase-map candidates for manuscript figure selection."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "gfx" / "candidate_phase_maps"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "figure.dpi": 150,
    }
)

PHASE = plt.cm.magma_r
PHASE_FORWARD = plt.cm.magma
GOLD = PHASE(0.08)
INK = "#1f2933"
GRID = "#eef1f5"
KEYS = ["beta_pi", "lambda_pi", "gamma"]
GAMMA = 0.12
TASK_LABELS = {
    "mackey_glass": "Mackey--Glass",
    "lorenz": "Lorenz",
    "narma10": "NARMA10",
    "sunspots_annual": "Sunspots",
}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=420, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["replicate"] = d["task"] + "__seed" + d["seed"].astype(str)
    d["val_rank_pct"] = d.groupby("replicate")["val_nmse"].rank(method="average", pct=True)
    d["test_rank_pct"] = d.groupby("replicate")["test_nmse"].rank(method="average", pct=True)
    return d


def smooth_grid(x, y, z, xmax, ymax, nx=260, ny=220):
    xi = np.linspace(0, xmax, nx)
    yi = np.linspace(0, ymax, ny)
    XI, YI = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (XI, YI), method="cubic")
    Zn = griddata((x, y), z, (XI, YI), method="nearest")
    Zs = np.where(np.isnan(Zi), Zn, Zi)
    for _ in range(2):
        P = np.pad(Zs, 1, mode="edge")
        Zs = (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] + 4 * P[1:-1, 1:-1]) / 8.0
    return XI, YI, Zs


def safe_contour(ax, XI, YI, Z, levels, **kwargs):
    valid = sorted({level for level in levels if float(np.nanmin(Z)) <= level <= float(np.nanmax(Z))})
    if valid:
        ax.contour(XI, YI, Z, levels=valid, **kwargs)


def style_axis(ax, title: str, show_labels: bool = True) -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=6)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#26323f")
    ax.tick_params(labelsize=8, length=2.2, width=0.65, color="#26323f", pad=2)
    if show_labels:
        ax.set_xlabel(r"$\beta/\pi$", fontsize=10)
        ax.set_ylabel(r"$\lambda/\pi$", fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def band_overlay(ax, scale: float = 1.0, selected_on: bool = True, alpha: float = 0.22) -> None:
    if len(primary_slice):
        cx = float(primary_slice.beta_pi.mean())
        cy = float(primary_slice.lambda_pi.mean())
        width = max(0.09, float(primary_slice.beta_pi.max() - primary_slice.beta_pi.min()) + 0.08)
        height = max(0.10, float(primary_slice.lambda_pi.max() - primary_slice.lambda_pi.min()) + 0.09)
        ax.add_patch(
            Ellipse(
                (cx, cy),
                width=width,
                height=height,
                facecolor=GOLD,
                edgecolor="white",
                linewidth=0.6 * scale,
                alpha=alpha,
                zorder=3,
            )
        )
        ax.scatter(
            primary_slice.beta_pi,
            primary_slice.lambda_pi,
            s=44 * scale,
            color=GOLD,
            edgecolor="white",
            linewidth=0.7 * scale,
            clip_on=False,
            zorder=4,
        )
        ax.scatter(
            primary_slice.beta_pi,
            primary_slice.lambda_pi,
            s=15 * scale,
            color=GOLD,
            edgecolor=INK,
            linewidth=0.3 * scale,
            clip_on=False,
            zorder=5,
        )
    if selected_on:
        ax.scatter(
            [selected["beta_pi"]],
            [selected["lambda_pi"]],
            marker="*",
            s=115 * scale,
            color=GOLD,
            edgecolor="white",
            linewidth=0.8 * scale,
            clip_on=False,
            zorder=6,
        )
        ax.scatter(
            [selected["beta_pi"]],
            [selected["lambda_pi"]],
            marker="*",
            s=62 * scale,
            color=GOLD,
            edgecolor=INK,
            linewidth=0.35 * scale,
            clip_on=False,
            zorder=7,
        )


def phase_surface(ax, agg, value, *, cmap, vmin, vmax, title, cbar_label, contour_levels=None, band=True):
    XI, YI, Z = smooth_grid(agg.beta_pi.values, agg.lambda_pi.values, agg[value].values, xmax, ymax)
    Zp = np.clip(Z, vmin, vmax)
    im = ax.imshow(
        Zp,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="bicubic",
        resample=True,
    )
    if contour_levels:
        safe_contour(ax, XI, YI, Zp, contour_levels[:1], colors=[GOLD], linewidths=[0.78], alpha=0.88)
        safe_contour(ax, XI, YI, Zp, contour_levels[1:], colors="white", linewidths=0.34, alpha=0.56)
    style_axis(ax, title)
    if band:
        band_overlay(ax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
    cbar.set_label(cbar_label, fontsize=9)
    return im


stats = json.loads((DATA / "phase_map_generalization_stats.json").read_text())
q = add_ranks(pd.read_csv(DATA / "qrc_seed_ensemble_grid.csv"))
xmax = float(q.beta_pi.max())
ymax = float(q.lambda_pi.max())
selected = stats["bands"][stats["primary_band"]]["medoid"]
membership = pd.read_csv(DATA / "phase_map_band_membership.csv")
primary = membership[(membership.p == 20) & np.isclose(membership.q, 0.7) & membership.in_band]
primary_slice = primary[np.isclose(primary.gamma, GAMMA)][["beta_pi", "lambda_pi"]]
qg = q[np.isclose(q.gamma, GAMMA)].copy()

# 1. Holdout phase map.
holdout = qg.groupby(["beta_pi", "lambda_pi"]).agg(mean_holdout_rank=("test_rank_pct", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(4.6, 3.6))
phase_surface(
    ax,
    holdout,
    "mean_holdout_rank",
    cmap=PHASE,
    vmin=0.05,
    vmax=0.80,
    title=r"Candidate 1: holdout rank map ($\gamma=0.12$)",
    cbar_label="holdout rank percentile",
    contour_levels=[0.20, 0.30, 0.45],
)
ax.text(0.02, 0.97, "band fixed by validation", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="white")
save(fig, "candidate_1_holdout_phase_map")

# 2. Band-frequency map.
freq = membership[(membership.p == 20) & np.isclose(membership.q, 0.7) & np.isclose(membership.gamma, GAMMA)].copy()
fig, ax = plt.subplots(figsize=(4.6, 3.6))
phase_surface(
    ax,
    freq,
    "top_frequency",
    cmap=PHASE_FORWARD,
    vmin=0.0,
    vmax=1.0,
    title=r"Candidate 2: top-20 frequency map ($\gamma=0.12$)",
    cbar_label=r"$\Pr[\mathrm{top}\ 20\%]$",
    contour_levels=[0.70, 0.50, 0.30],
)
save(fig, "candidate_2_band_frequency_map")

# 3. Seed-stability / variance map.
seed_mean = qg.groupby(["beta_pi", "lambda_pi", "seed"]).agg(seed_mean_rank=("val_rank_pct", "mean")).reset_index()
stability = seed_mean.groupby(["beta_pi", "lambda_pi"]).agg(seed_rank_std=("seed_mean_rank", "std")).reset_index()
fig, ax = plt.subplots(figsize=(4.6, 3.6))
phase_surface(
    ax,
    stability,
    "seed_rank_std",
    cmap=PHASE,
    vmin=0.00,
    vmax=0.22,
    title=r"Candidate 3: seed-stability map ($\gamma=0.12$)",
    cbar_label="std. of seed-mean rank",
    contour_levels=[0.06, 0.10, 0.16],
)
ax.text(0.02, 0.97, "bright = more stable", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="white")
save(fig, "candidate_3_seed_stability_map")

# 4. Base-minus-ablation delta maps.
abl = pd.read_csv(DATA / "qrc_phase_ablation_slice_grid.csv")
abl["replicate"] = abl["variant"] + "__" + abl["task"] + "__seed" + abl["seed"].astype(str)
abl["val_rank_pct"] = abl.groupby("replicate")["val_nmse"].rank(method="average", pct=True)
base = (
    abl[abl.variant == "base_amplitude_rxzz"]
    .groupby(["beta_pi", "lambda_pi"])
    .agg(base_rank=("val_rank_pct", "mean"))
    .reset_index()
)
delta_variants = [
    ("gamma0_amplitude", r"$\gamma=0$"),
    ("mixer_none", "no recurrent mixing"),
    ("dephasing", "dephasing"),
]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), sharex=True, sharey=True)
last_im = None
for ax, (variant, label) in zip(axes, delta_variants):
    v = abl[abl.variant == variant].groupby(["beta_pi", "lambda_pi"]).agg(ablation_rank=("val_rank_pct", "mean")).reset_index()
    d = base.merge(v, on=["beta_pi", "lambda_pi"])
    d["rank_loss"] = np.maximum(d["ablation_rank"] - d["base_rank"], 0.0)
    XI, YI, Z = smooth_grid(d.beta_pi.values, d.lambda_pi.values, d.rank_loss.values, xmax, ymax)
    Zp = np.clip(Z, 0.0, 0.45)
    last_im = ax.imshow(
        Zp,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap=PHASE_FORWARD,
        vmin=0.0,
        vmax=0.45,
        aspect="auto",
        interpolation="bicubic",
        resample=True,
    )
    safe_contour(ax, XI, YI, Zp, [0.10, 0.20, 0.30], colors="white", linewidths=[0.42, 0.34, 0.28], alpha=0.62)
    style_axis(ax, label, show_labels=True)
    if ax is not axes[0]:
        ax.set_ylabel("")
    band_overlay(ax, scale=0.65, selected_on=False, alpha=0.20)
fig.suptitle("Candidate 4: rank loss when mechanism is ablated", fontsize=13, color=INK, y=1.03)
cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.030, pad=0.018)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
cbar.set_label("ablation rank loss", fontsize=9)
save(fig, "candidate_4_ablation_delta_maps")

# 5. Per-task phase maps.
task_order = ["mackey_glass", "lorenz", "narma10", "sunspots_annual"]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True, sharey=True)
for ax, task in zip(axes.ravel(), task_order):
    d = qg[qg.task == task].groupby(["beta_pi", "lambda_pi"]).agg(mean_rank=("val_rank_pct", "mean")).reset_index()
    XI, YI, Z = smooth_grid(d.beta_pi.values, d.lambda_pi.values, d.mean_rank.values, xmax, ymax)
    Zp = np.clip(Z, 0.05, 0.80)
    im = ax.imshow(Zp, origin="lower", extent=[0, xmax, 0, ymax], cmap=PHASE, vmin=0.05, vmax=0.80, aspect="auto", interpolation="bicubic")
    safe_contour(ax, XI, YI, Zp, [0.20], colors=[GOLD], linewidths=[0.70], alpha=0.84)
    safe_contour(ax, XI, YI, Zp, [0.30, 0.45], colors="white", linewidths=0.32, alpha=0.58)
    style_axis(ax, TASK_LABELS[task])
    band_overlay(ax, scale=0.75, selected_on=True, alpha=0.18)
for ax in axes[:, 1]:
    ax.set_ylabel("")
for ax in axes[0, :]:
    ax.set_xlabel("")
fig.suptitle(r"Candidate 5: per-task validation phase maps ($\gamma=0.12$)", fontsize=13, color=INK, y=1.01)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.018)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
cbar.set_label("validation rank percentile", fontsize=9)
save(fig, "candidate_5_per_task_phase_maps")

print(f"Wrote candidate figures to {OUT}")
