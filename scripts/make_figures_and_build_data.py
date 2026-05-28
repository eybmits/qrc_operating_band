#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator, griddata
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GFX = ROOT / "paper" / "gfx"
GFX.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
    }
)

PHASE_CMAP = plt.cm.magma_r
PHASE_FORWARD = plt.cm.magma
PHASE_GOLD = PHASE_CMAP(0.08)
PHASE_AMBER = PHASE_CMAP(0.18)
PHASE_CORAL = PHASE_CMAP(0.32)
PHASE_ROSE = PHASE_CMAP(0.46)
PHASE_VIOLET = PHASE_CMAP(0.82)
INK = "#1f2933"
GRID = "#d9dee7"
RANK_VMIN = 0.05
RANK_VMAX = 0.80

KEYS = ["beta_pi", "lambda_pi", "gamma"]
TASK_LABELS = {
    "mackey_glass": "MG",
    "narma10": "NARMA10",
    "lorenz": "Lorenz",
    "sunspots_annual": "Sunspots",
}


def savefig_dual(fig, stem, aliases=()):
    fig.savefig(GFX / f"{stem}.png", dpi=360, bbox_inches="tight")
    fig.savefig(GFX / f"{stem}.pdf", bbox_inches="tight")
    for alias in aliases:
        fig.savefig(GFX / f"{alias}.png", dpi=360, bbox_inches="tight")
        fig.savefig(GFX / f"{alias}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_ranks(df: pd.DataFrame, by=("task", "seed")) -> pd.DataFrame:
    d = df.copy()
    d["replicate"] = d["task"] + "__seed" + d["seed"].astype(str)
    d["val_rank_pct"] = d.groupby(list(by))["val_nmse"].rank(method="average", pct=True)
    return d


def smooth_grid(x, y, z, xmax, ymax, nx=220, ny=180):
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


def retention_sigmoid(x, x0, k, a):
    with np.errstate(over="ignore"):
        return 100.0 / (1.0 + np.exp(-k * (x - x0))) ** a


def smooth_retention_curve(x, y):
    x = np.asarray(x, dtype=float)
    y = np.maximum.accumulate(np.asarray(y, dtype=float))
    dense = np.linspace(float(x.min()), float(x.max()), 700)
    starts = [
        (22.0, 0.16, 0.6),
        (28.0, 0.18, 0.5),
        (32.0, 0.12, 1.0),
        (55.0, 0.14, 1.0),
        (60.0, 0.18, 1.4),
    ]
    best = None
    for p0 in starts:
        try:
            popt, _ = curve_fit(
                retention_sigmoid,
                x,
                y,
                p0=p0,
                bounds=([0.0, 0.01, 0.2], [100.0, 1.0, 5.0]),
                maxfev=20000,
            )
            err = float(np.sqrt(np.mean((retention_sigmoid(x, *popt) - y) ** 2)))
            if best is None or err < best[0]:
                best = (err, popt)
        except (RuntimeError, ValueError):
            continue
    if best is None:
        vals = PchipInterpolator(x, y)(dense)
    else:
        vals = retention_sigmoid(dense, *best[1])
    vals = np.maximum.accumulate(np.clip(vals, 0.0, 100.0))
    vals[-1] = 100.0
    return dense, vals


def polish_phase_axis(ax, labelsize=6.5, show_ticks=True):
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#24313d")
    ax.set_facecolor("#111827")
    ax.tick_params(labelsize=labelsize, length=2.0, width=0.55, color="#24313d", pad=1.5)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_band_overlay(ax, core, selected=None, marker_scale=1.0):
    if core is not None and len(core):
        cx = float(core.beta_pi.mean())
        cy = float(core.lambda_pi.mean())
        width = max(0.085, float(core.beta_pi.max() - core.beta_pi.min()) + 0.075)
        height = max(0.095, float(core.lambda_pi.max() - core.lambda_pi.min()) + 0.085)
        halo = Ellipse(
            (cx, cy),
            width=width,
            height=height,
            facecolor=PHASE_GOLD,
            edgecolor="white",
            linewidth=0.45 * marker_scale,
            alpha=0.20,
            zorder=3,
        )
        ax.add_patch(halo)
        ax.scatter(
            core.beta_pi,
            core.lambda_pi,
            s=34 * marker_scale,
            color=PHASE_GOLD,
            edgecolor="white",
            linewidth=0.55 * marker_scale,
            alpha=0.97,
            clip_on=False,
            zorder=4,
        )
        ax.scatter(
            core.beta_pi,
            core.lambda_pi,
            s=12 * marker_scale,
            facecolor=PHASE_GOLD,
            edgecolor=INK,
            linewidth=0.25 * marker_scale,
            clip_on=False,
            zorder=5,
        )
    if selected is not None:
        ax.scatter(
            [selected["beta_pi"]],
            [selected["lambda_pi"]],
            marker="*",
            s=74 * marker_scale,
            color=PHASE_GOLD,
            edgecolor="white",
            linewidth=0.65 * marker_scale,
            clip_on=False,
            zorder=6,
        )
        ax.scatter(
            [selected["beta_pi"]],
            [selected["lambda_pi"]],
            marker="*",
            s=43 * marker_scale,
            color=PHASE_GOLD,
            edgecolor=INK,
            linewidth=0.28 * marker_scale,
            clip_on=False,
            zorder=7,
        )


def plot_rank_map(ax, d, title, xmax, ymax, selected=None, core=None, labelsize=6.5, highlight=False, show_ticks=True):
    agg = d.groupby(["beta_pi", "lambda_pi"]).agg(mean_rank=("val_rank_pct", "mean")).reset_index()
    XI, YI, Zs = smooth_grid(agg.beta_pi.values, agg.lambda_pi.values, agg.mean_rank.values, xmax, ymax)
    Zp = np.clip(Zs, RANK_VMIN, RANK_VMAX)
    im = ax.imshow(
        Zp,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap=PHASE_CMAP,
        vmin=RANK_VMIN,
        vmax=RANK_VMAX,
        aspect="auto",
        interpolation="bicubic",
        resample=True,
    )
    safe_contour(ax, XI, YI, Zp, [0.20], colors=[PHASE_GOLD], linewidths=[0.75], alpha=0.86)
    safe_contour(ax, XI, YI, Zp, [0.30, 0.45], colors="white", linewidths=[0.36, 0.28], alpha=0.58)
    plot_band_overlay(ax, core, selected=selected, marker_scale=max(0.75, labelsize / 7.0))
    ax.set_title(title, fontsize=labelsize + 0.8, pad=2.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_box_aspect(1)
    polish_phase_axis(ax, labelsize=labelsize, show_ticks=show_ticks)
    if highlight:
        for spine in ax.spines.values():
            spine.set_color(PHASE_GOLD)
            spine.set_linewidth(1.0)
    return im


def core_points_at_slice(d, p=0.20, q=0.70):
    parts = []
    for _, g in d.groupby("replicate"):
        h = g.copy()
        h["top"] = h.val_rank_pct <= p
        parts.append(h)
    return pd.concat(parts).groupby(["beta_pi", "lambda_pi"]).top.mean().reset_index().query("top >= @q")


stats_path = DATA / "phase_map_generalization_stats.json"
if not stats_path.exists():
    raise FileNotFoundError(f"Missing {stats_path}; run scripts/analyze_phase_map_generalization.py first.")
stats = json.loads(stats_path.read_text())

q = add_ranks(pd.read_csv(DATA / "qrc_seed_ensemble_grid.csv"))
all_tasks = sorted(q.task.unique())
gamma_star = 0.12
qg = q[np.isclose(q.gamma, gamma_star)]
xmax = float(q.beta_pi.max())
ymax = float(q.lambda_pi.max())
selected = stats["bands"][stats["primary_band"]]["medoid"]
primary_members = pd.read_csv(DATA / "phase_map_band_membership.csv")
primary_band = primary_members[(primary_members.p == 20) & np.isclose(primary_members.q, 0.7) & primary_members.in_band]
primary_slice = primary_band[np.isclose(primary_band.gamma, gamma_star)][["beta_pi", "lambda_pi"]]

# Figure 1: compact leave-one-task-out phase maps.
panels = [("(a) all tasks used", all_tasks)] + [
    (f"({chr(ord('b') + i)}) {TASK_LABELS[t]} removed", [x for x in all_tasks if x != t])
    for i, t in enumerate(["mackey_glass", "narma10", "lorenz", "sunspots_annual"])
]
fig, axes = plt.subplots(1, 5, figsize=(7.25, 2.64), sharey=True)
for idx, (ax, (title, tasks)) in enumerate(zip(axes, panels)):
    d = qg[qg.task.isin(tasks)].copy()
    core = core_points_at_slice(d)
    im = plot_rank_map(ax, d, title, xmax, ymax, selected=selected, core=core, labelsize=6)
    ax.set_xlabel(r"$\beta/\pi$", fontsize=7)
    ticks = [0.0, 0.25] if idx < len(axes) - 1 else [0.0, 0.25, 0.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])
axes[0].set_ylabel(r"$\lambda/\pi$", fontsize=7.5)
fig.subplots_adjust(left=0.035, right=0.955, bottom=0.19, top=0.91, wspace=0.06)
fig.canvas.draw()
right_box = axes[-1].get_position()
cax = fig.add_axes([right_box.x1 + 0.0075, right_box.y0, 0.0085, right_box.height])
cbar = fig.colorbar(im, cax=cax)
cbar.outline.set_visible(False)
cbar.set_ticks([0.2, 0.4, 0.6, 0.8])
cbar.set_label("rank percentile", fontsize=6.2, labelpad=3)
cbar.ax.tick_params(labelsize=5.8, length=1.8, width=0.5)
savefig_dual(fig, "fig1_short_phase_maps")

# Figure 2: damping-slice atlas for the same validation-ranked phase grid.
broad_band = primary_members[
    (primary_members.p == 30) & np.isclose(primary_members.q, 0.7) & primary_members.in_band
]
gamma_values = [0.0, 0.01, 0.05, 0.12, 0.22, 0.30]
fig, axes = plt.subplots(1, 5, figsize=(7.25, 2.64), sharey=True)
gamma_im = None
for idx, (ax, gamma) in enumerate(zip(axes, gamma_values)):
    d = q[np.isclose(q.gamma, gamma)]
    agg = d.groupby(["beta_pi", "lambda_pi"]).agg(mean_rank=("val_rank_pct", "mean")).reset_index()
    XI, YI, Zs = smooth_grid(agg.beta_pi.values, agg.lambda_pi.values, agg.mean_rank.values, xmax, ymax)
    Zp = np.clip(Zs, RANK_VMIN, RANK_VMAX)
    gamma_im = ax.imshow(
        Zp,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap=PHASE_CMAP,
        vmin=RANK_VMIN,
        vmax=RANK_VMAX,
        aspect="auto",
        interpolation="bicubic",
        resample=True,
    )
    safe_contour(ax, XI, YI, Zp, [0.20], colors=[PHASE_GOLD], linewidths=[0.75], alpha=0.88)
    safe_contour(ax, XI, YI, Zp, [0.30, 0.45], colors="white", linewidths=[0.36, 0.28], alpha=0.58)

    broad_slice = broad_band[np.isclose(broad_band.gamma, gamma)]
    core_slice = primary_band[np.isclose(primary_band.gamma, gamma)]
    if len(broad_slice):
        ax.scatter(
            broad_slice.beta_pi,
            broad_slice.lambda_pi,
            s=26,
            facecolors=(1.0, 0.86, 0.43, 0.28),
            edgecolors="white",
            linewidths=0.75,
            zorder=4,
        )
        ax.scatter(
            broad_slice.beta_pi,
            broad_slice.lambda_pi,
            s=13,
            facecolors="none",
            edgecolors="#8a6200",
            linewidths=0.6,
            zorder=5,
        )
    if len(core_slice):
        ax.scatter(
            core_slice.beta_pi,
            core_slice.lambda_pi,
            s=54,
            facecolors="#fff0a6",
            edgecolors="white",
            linewidths=0.9,
            zorder=6,
        )
        ax.scatter(
            [selected["beta_pi"]],
            [selected["lambda_pi"]],
            marker="*",
            s=122,
            facecolors="white",
            edgecolors=INK,
            linewidths=0.6,
            zorder=7,
        )

    ax.set_title(rf"({chr(97 + idx)}) $\gamma={gamma:g}$", fontsize=6.8, pad=2.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_box_aspect(1)
    ticks = [0.0, 0.25] if idx < len(axes) - 1 else [0.0, 0.25, 0.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks], fontsize=6)
    ax.set_yticks([0, 0.2, 0.4])
    if idx == 0:
        ax.set_ylabel(r"$\lambda/\pi$", fontsize=7.5)
        ax.set_yticklabels(["0", "0.2", "0.4"], fontsize=6)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(r"$\beta/\pi$", fontsize=7, labelpad=1)
    ax.tick_params(length=2.0, pad=1.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

fig.subplots_adjust(left=0.035, right=0.955, bottom=0.19, top=0.91, wspace=0.06)
fig.canvas.draw()
right_box = axes[-1].get_position()
cax = fig.add_axes([right_box.x1 + 0.0075, right_box.y0, 0.0085, right_box.height])
cbar = fig.colorbar(gamma_im, cax=cax)
cbar.set_label("rank percentile", fontsize=6.2, color=INK, labelpad=3)
cbar.set_ticks([0.2, 0.4, 0.6, 0.8])
cbar.ax.tick_params(labelsize=5.8, length=1.8, width=0.5)
cbar.outline.set_visible(False)
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="none",
        markersize=9.5,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=0.7,
        label=r"$B_{20,0.7}$ medoid",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=6.8,
        markerfacecolor="#fff0a6",
        markeredgecolor="white",
        markeredgewidth=0.9,
        label=r"$B_{20,0.7}$ strict core",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.8,
        markerfacecolor="none",
        markeredgecolor="#8a6200",
        markeredgewidth=0.8,
        label=r"$B_{30,0.7}$ broader band",
    ),
]
axes[-1].legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.995, 0.985),
    frameon=True,
    framealpha=0.78,
    facecolor="white",
    edgecolor="none",
    fontsize=4.6,
    handletextpad=0.22,
    borderpad=0.18,
    labelspacing=0.12,
)
savefig_dual(fig, "gamma_regime_slices_only")

# Compact single-column version for the four-page workshop paper.
compact_gamma_values = [0.01, 0.12, 0.22, 0.30]
fig, axes = plt.subplots(2, 2, figsize=(3.34, 3.36), sharex=True, sharey=True)
gamma_im = None
for idx, (ax, gamma) in enumerate(zip(axes.ravel(), compact_gamma_values)):
    d = q[np.isclose(q.gamma, gamma)]
    agg = d.groupby(["beta_pi", "lambda_pi"]).agg(mean_rank=("val_rank_pct", "mean")).reset_index()
    XI, YI, Zs = smooth_grid(agg.beta_pi.values, agg.lambda_pi.values, agg.mean_rank.values, xmax, ymax)
    Zp = np.clip(Zs, RANK_VMIN, RANK_VMAX)
    gamma_im = ax.imshow(
        Zp,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap=PHASE_CMAP,
        vmin=RANK_VMIN,
        vmax=RANK_VMAX,
        aspect="auto",
        interpolation="bicubic",
        resample=True,
    )
    safe_contour(ax, XI, YI, Zp, [0.20], colors=[PHASE_GOLD], linewidths=[0.65], alpha=0.9)
    safe_contour(ax, XI, YI, Zp, [0.30, 0.45], colors="white", linewidths=[0.30, 0.24], alpha=0.55)

    broad_slice = broad_band[np.isclose(broad_band.gamma, gamma)]
    core_slice = primary_band[np.isclose(primary_band.gamma, gamma)]
    if len(broad_slice):
        ax.scatter(
            broad_slice.beta_pi,
            broad_slice.lambda_pi,
            s=18,
            facecolors=(1.0, 0.86, 0.43, 0.26),
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        ax.scatter(
            broad_slice.beta_pi,
            broad_slice.lambda_pi,
            s=8,
            facecolors="none",
            edgecolors="#8a6200",
            linewidths=0.45,
            zorder=5,
        )
    if len(core_slice):
        ax.scatter(
            core_slice.beta_pi,
            core_slice.lambda_pi,
            s=31,
            facecolors="#fff0a6",
            edgecolors="white",
            linewidths=0.7,
            zorder=6,
        )
        ax.scatter(
            [selected["beta_pi"]],
            [selected["lambda_pi"]],
            marker="*",
            s=70,
            facecolors="white",
            edgecolors=INK,
            linewidths=0.5,
            zorder=7,
        )

    ax.set_title(rf"({chr(97 + idx)}) $\gamma={gamma:g}$", fontsize=7.0, pad=1.8, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_box_aspect(1)
    ax.set_xticks([0.0, 0.25, 0.5])
    ax.set_xticklabels(["0", "0.25", "0.5"], fontsize=6.0)
    ax.set_yticks([0.0, 0.2, 0.4])
    ax.set_yticklabels(["0", "0.2", "0.4"], fontsize=6.0)
    ax.tick_params(length=1.8, pad=1.0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\lambda/\pi$", fontsize=7.0, labelpad=1.0)
for ax in axes[-1, :]:
    ax.set_xlabel(r"$\beta/\pi$", fontsize=7.0, labelpad=1.0)

fig.subplots_adjust(left=0.12, right=0.89, bottom=0.10, top=0.93, wspace=0.08, hspace=0.20)
fig.canvas.draw()
top_box = axes[0, 1].get_position()
bottom_box = axes[1, 1].get_position()
cax = fig.add_axes([top_box.x1 + 0.020, bottom_box.y0, 0.026, top_box.y1 - bottom_box.y0])
cbar = fig.colorbar(gamma_im, cax=cax)
cbar.set_label("rank percentile\nbright = better", fontsize=6.0, color=INK, labelpad=2.5)
cbar.set_ticks([0.2, 0.4, 0.6, 0.8])
cbar.ax.tick_params(labelsize=5.8, length=1.6, width=0.45)
cbar.outline.set_visible(False)
savefig_dual(fig, "fig4_gamma_slices_compact")

# Figure 3: validation-band frequency and mechanism-sensitive ablation loss.
abl = pd.read_csv(DATA / "qrc_phase_ablation_slice_grid.csv")
abl["replicate"] = abl["variant"] + "__" + abl["task"] + "__seed" + abl["seed"].astype(str)
abl["val_rank_pct"] = abl.groupby("replicate")["val_nmse"].rank(method="average", pct=True)
fig, axes = plt.subplots(2, 2, figsize=(3.34, 3.36), sharex=True, sharey=True)

freq = primary_members[
    (primary_members.p == 20) & np.isclose(primary_members.q, 0.7) & np.isclose(primary_members.gamma, gamma_star)
].copy()
ax = axes[0, 0]
XI, YI, Zs = smooth_grid(freq.beta_pi.values, freq.lambda_pi.values, freq.top_frequency.values, xmax, ymax, nx=220, ny=180)
Zp = np.clip(Zs, 0.0, 1.0)
freq_im = ax.imshow(
    Zp,
    origin="lower",
    extent=[0, xmax, 0, ymax],
    cmap=PHASE_FORWARD,
    vmin=0.0,
    vmax=1.0,
    aspect="auto",
    interpolation="bicubic",
    resample=True,
)
safe_contour(ax, XI, YI, Zp, [0.70], colors=[PHASE_GOLD], linewidths=[0.64], alpha=0.9)
safe_contour(ax, XI, YI, Zp, [0.30, 0.50], colors="white", linewidths=[0.28, 0.34], alpha=0.62)
plot_band_overlay(ax, primary_slice, selected=selected, marker_scale=0.52)
ax.set_title("(a) band frequency", fontsize=7.0, color=INK, pad=1.8)
ax.set_ylabel(r"$\lambda/\pi$", fontsize=7.0, labelpad=1.0)
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.set_box_aspect(1)
polish_phase_axis(ax, labelsize=5.6)
ax.text(
    0.04,
    0.95,
    r"$B_{20,0.7}$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=5.7,
    color="white",
    bbox=dict(boxstyle="round,pad=0.18", facecolor="#111827", edgecolor="none", alpha=0.55),
)
freq_cax = ax.inset_axes([0.56, 0.862, 0.34, 0.035])
freq_cbar = fig.colorbar(freq_im, cax=freq_cax, orientation="horizontal")
freq_cbar.outline.set_edgecolor("white")
freq_cbar.outline.set_linewidth(0.35)
freq_cbar.set_ticks([0.0, 1.0])
freq_cbar.ax.tick_params(labelsize=4.2, colors="white", length=1.1, width=0.32, pad=0.3)
freq_cbar.ax.text(
    0.5,
    1.75,
    "selection freq.",
    transform=freq_cbar.ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=4.3,
    color="white",
)
ax.text(0.96, 0.06, "bright = frequent", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.6, color="white", alpha=0.9)

base = (
    abl[abl.variant == "base_amplitude_rxzz"]
    .groupby(["beta_pi", "lambda_pi"])
    .agg(base_rank=("val_rank_pct", "mean"))
    .reset_index()
)
delta_variants = [
    ("gamma0_amplitude", r"(b) remove damping"),
    ("mixer_none", "(c) remove mixing"),
    ("dephasing", "(d) dephasing channel"),
]
delta_axes = []
delta_im = None
for i, (variant, title) in enumerate(delta_variants, start=1):
    ax = axes.ravel()[i]
    delta_axes.append(ax)
    v = (
        abl[abl.variant == variant]
        .groupby(["beta_pi", "lambda_pi"])
        .agg(ablation_rank=("val_rank_pct", "mean"))
        .reset_index()
    )
    d = base.merge(v, on=["beta_pi", "lambda_pi"])
    d["rank_loss"] = np.maximum(d["ablation_rank"] - d["base_rank"], 0.0)
    XI, YI, Zs = smooth_grid(d.beta_pi.values, d.lambda_pi.values, d.rank_loss.values, xmax, ymax, nx=220, ny=180)
    Zp = np.clip(Zs, 0.0, 0.45)
    delta_im = ax.imshow(
        Zp,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap=PHASE_CMAP,
        vmin=0.0,
        vmax=0.45,
        aspect="auto",
        interpolation="bicubic",
        resample=True,
    )
    safe_contour(ax, XI, YI, Zp, [0.10, 0.20, 0.30], colors="white", linewidths=[0.24, 0.30, 0.36], alpha=0.66)
    plot_band_overlay(ax, primary_slice, selected=None, marker_scale=0.48)
    ax.set_title(title, fontsize=7.0, color=INK, pad=1.8)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_box_aspect(1)
    polish_phase_axis(ax, labelsize=5.6)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\lambda/\pi$", fontsize=7.0, labelpad=1.0)
for ax in axes[-1, :]:
    ax.set_xlabel(r"$\beta/\pi$", fontsize=7.0, labelpad=1.0)
for ax in axes.ravel():
    ax.set_xticks([0.0, 0.25, 0.5])
    ax.set_xticklabels(["0", "0.25", "0.5"], fontsize=5.6)
    ax.set_yticks([0.0, 0.2, 0.4])
    ax.set_yticklabels(["0", "0.2", "0.4"], fontsize=5.6)
    ax.tick_params(length=1.8, pad=1.0)
fig.subplots_adjust(left=0.12, right=0.89, bottom=0.10, top=0.93, wspace=0.08, hspace=0.20)
fig.canvas.draw()
top_box = axes[0, 1].get_position()
bottom_box = axes[1, 1].get_position()
cax = fig.add_axes([top_box.x1 + 0.020, bottom_box.y0, 0.026, top_box.y1 - bottom_box.y0])
cbar = fig.colorbar(delta_im, cax=cax)
cbar.outline.set_visible(False)
cbar.set_ticks([0.0, 0.15, 0.30, 0.45])
cbar.ax.tick_params(labelsize=5.8, length=1.6, width=0.45, pad=1.2)
cbar.set_label("rank loss\n(dark = worse)", fontsize=6.0, labelpad=2.5)
savefig_dual(fig, "fig2_short_evidence")

# Figure 4: diagnostic memory map, global memory relation, and screening retention.
screen = pd.read_csv(DATA / "screening_retention_recomputed_intrinsic_diagnostics.csv")
diag = pd.read_csv(DATA / "qrc_real_current_intrinsic_diagnostics_with_perf.csv")
diag["logMC"] = np.log1p(diag["MC"])
spearman = pd.read_csv(DATA / "qrc_real_current_diagnostic_spearman_named.csv")
fig = plt.figure(figsize=(7.25, 2.28))
gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 1.15, 1.15], wspace=0.22)
axes = np.array([fig.add_subplot(gs[0, i]) for i in range(3)])

memory_slice = (
    diag[np.isclose(diag.gamma, gamma_star)]
    .groupby(["beta_pi", "lambda_pi"])
    .agg(logMC=("logMC", "mean"))
    .reset_index()
)
ax = axes[0]
XI, YI, Zs = smooth_grid(
    memory_slice.beta_pi.values,
    memory_slice.lambda_pi.values,
    memory_slice.logMC.values,
    xmax,
    ymax,
    nx=220,
    ny=180,
)
memory_max = float(np.nanpercentile(memory_slice.logMC, 99))
Zp = np.clip(Zs, 0.0, memory_max)
memory_im = ax.imshow(
    Zp,
    origin="lower",
    extent=[0, xmax, 0, ymax],
    cmap=PHASE_FORWARD,
    vmin=0.0,
    vmax=memory_max,
    aspect="auto",
    interpolation="bicubic",
    resample=True,
)
memory_levels = [float(np.nanpercentile(memory_slice.logMC, p)) for p in [55, 75, 90]]
safe_contour(ax, XI, YI, Zp, memory_levels, colors="white", linewidths=[0.25, 0.35, 0.45], alpha=0.62)
plot_band_overlay(ax, primary_slice, selected=selected, marker_scale=0.62)
ax.set_title("(a) memory map", fontsize=8.8, pad=3)
ax.set_xlabel(r"$\beta/\pi$", fontsize=7.8)
ax.set_ylabel(r"$\lambda/\pi$", fontsize=7.8)
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.set_xticks([0.0, 0.25, 0.5])
ax.set_yticks([0.0, 0.2, 0.4])
ax.set_box_aspect(1)
polish_phase_axis(ax, labelsize=6.8)
memory_cax = ax.inset_axes([0.54, 0.07, 0.38, 0.04])
memory_cbar = fig.colorbar(memory_im, cax=memory_cax, orientation="horizontal")
memory_cbar.outline.set_edgecolor("white")
memory_cbar.outline.set_linewidth(0.35)
memory_cbar.set_ticks([0.0, memory_max])
memory_cbar.ax.set_xticklabels(["0", "high"])
memory_cbar.ax.tick_params(labelsize=4.8, colors="white", length=1.2, width=0.35, pad=0.4)
memory_cbar.ax.text(
    0.5,
    1.70,
    r"$\log(1+MC)$",
    transform=memory_cbar.ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=5.0,
    color="white",
)

dead_mc_threshold = 0.1
active_mc = diag[diag.MC > dead_mc_threshold]
axes[1].scatter(active_mc.logMC, active_mc.mean_val_rank, s=8.5, alpha=0.24, color=PHASE_ROSE, edgecolor="none")
z = np.polyfit(active_mc.logMC, active_mc.mean_val_rank, 1)
xs = np.linspace(active_mc.logMC.min(), active_mc.logMC.max(), 220)
axes[1].plot(xs, z[0] * xs + z[1], color="black", lw=1.0)
axes[1].set_title("(b) memory predicts rank", fontsize=8.8, pad=3)
axes[1].set_xlabel(r"$\log(1+\mathrm{MC})$", fontsize=7.8)
axes[1].set_ylabel("validation rank", fontsize=7.8, labelpad=1.0)

screen_x = screen["budget_pct"].to_numpy(dtype=float)
for col, color in [("IPCtot", PHASE_GOLD), ("MC", PHASE_ROSE), ("Vfeat", PHASE_VIOLET), ("random", "#9ca3af")]:
    screen_y = np.maximum.accumulate(screen[col].to_numpy(dtype=float))
    if col == "random":
        axes[2].plot(screen_x, screen_y, label=col, color=color, lw=1.15)
    else:
        screen_dense, smooth_y = smooth_retention_curve(screen_x, screen_y)
        axes[2].plot(screen_dense, smooth_y, label=col, color=color, lw=1.65, solid_capstyle="round")
axes[2].set_title("(c) screening retention", fontsize=8.8, pad=3)
axes[2].set_xlabel("budget (%)", fontsize=7.8)
axes[2].set_ylabel("retained (%)", fontsize=7.8)
axes[2].set_xlim(5, 100)
axes[2].set_ylim(-4, 104)
axes[2].legend(
    frameon=False,
    fontsize=6.9,
    loc="lower right",
    handlelength=1.5,
    borderaxespad=0.25,
)
for i, ax in enumerate(axes):
    if i > 0:
        ax.grid(color=GRID, linewidth=0.42, alpha=0.68)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.8, length=2.0, width=0.55, pad=1.5)
fig.subplots_adjust(left=0.058, right=0.988, bottom=0.25, top=0.86)
savefig_dual(fig, "fig3_memory_capacity_screens")

summary = {
    "primary_band": stats["primary_band"],
    "primary_band_size": stats["bands"][stats["primary_band"]]["size"],
    "primary_band_components": stats["bands"][stats["primary_band"]]["components"],
    "representative_beta_pi": selected["beta_pi"],
    "representative_lambda_pi": selected["lambda_pi"],
    "representative_gamma": selected["gamma"],
    "leave_one_task_mean_rank": stats["leave_one_task_out"]["mean"],
    "leave_one_task_ci95_low": stats["leave_one_task_out"]["ci95_low"],
    "leave_one_task_ci95_high": stats["leave_one_task_out"]["ci95_high"],
    "leave_one_seed_mean_rank": stats["leave_one_seed_out"]["mean"],
    "leave_one_seed_ci95_low": stats["leave_one_seed_out"]["ci95_low"],
    "leave_one_seed_ci95_high": stats["leave_one_seed_out"]["ci95_high"],
    "holdout_band_rank": stats["holdout_performance"][-1]["band_holdout_rank_mean"],
    "holdout_band_nmse": stats["holdout_performance"][-1]["band_holdout_nmse_mean"],
    "holdout_medoid_nmse": stats["holdout_performance"][-1]["medoid_holdout_nmse_mean"],
    "qrc_mc_spearman": float(spearman.set_index("metric").loc["MC", "spearman_vs_val_rank"]),
    "qrc_ipctot_spearman": float(spearman.set_index("metric").loc["IPCtot", "spearman_vs_val_rank"]),
}
(DATA / "final_summary_numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
