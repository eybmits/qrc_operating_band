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
    chunks = []
    for _, g in d.groupby("replicate"):
        h = g.copy()
        h["top"] = h.val_rank_pct <= p
        chunks.append(h)
    return pd.concat(chunks).groupby(["beta_pi", "lambda_pi"]).top.mean().reset_index().query("top >= @q")


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

# Figure 2: validation-band frequency and mechanism-sensitive ablation loss.
abl = pd.read_csv(DATA / "qrc_phase_ablation_slice_grid.csv")
abl["replicate"] = abl["variant"] + "__" + abl["task"] + "__seed" + abl["seed"].astype(str)
abl["val_rank_pct"] = abl.groupby("replicate")["val_nmse"].rank(method="average", pct=True)
fig = plt.figure(figsize=(7.25, 2.64))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.18)

freq = primary_members[
    (primary_members.p == 20) & np.isclose(primary_members.q, 0.7) & np.isclose(primary_members.gamma, gamma_star)
].copy()
ax = fig.add_subplot(gs[0, 0])
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
safe_contour(ax, XI, YI, Zp, [0.70], colors=[PHASE_GOLD], linewidths=[0.75], alpha=0.9)
safe_contour(ax, XI, YI, Zp, [0.30, 0.50], colors="white", linewidths=[0.32, 0.42], alpha=0.62)
plot_band_overlay(ax, primary_slice, selected=selected, marker_scale=0.62)
ax.set_title("(a) validation-band frequency", fontsize=7.8, color=INK, pad=3.5)
ax.set_xlabel(r"$\beta/\pi$", fontsize=6.4)
ax.set_ylabel(r"$\lambda/\pi$", fontsize=6.4)
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.set_box_aspect(1)
polish_phase_axis(ax, labelsize=5.8)
ax.text(
    0.04,
    0.95,
    r"$B_{20,0.7}$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=6.2,
    color="white",
    bbox=dict(boxstyle="round,pad=0.18", facecolor="#111827", edgecolor="none", alpha=0.55),
)
freq_cax = ax.inset_axes([0.60, 0.865, 0.32, 0.035])
freq_cbar = fig.colorbar(freq_im, cax=freq_cax, orientation="horizontal")
freq_cbar.outline.set_edgecolor("white")
freq_cbar.outline.set_linewidth(0.35)
freq_cbar.set_ticks([0.0, 1.0])
freq_cbar.ax.tick_params(labelsize=4.6, colors="white", length=1.2, width=0.35, pad=0.4)
freq_cbar.ax.text(
    0.5,
    1.75,
    "selection freq.",
    transform=freq_cbar.ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=4.8,
    color="white",
)
ax.text(0.96, 0.06, "bright = frequent", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color="white", alpha=0.9)

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
    ax = fig.add_subplot(gs[0, i])
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
    safe_contour(ax, XI, YI, Zp, [0.10, 0.20, 0.30], colors="white", linewidths=[0.28, 0.36, 0.44], alpha=0.66)
    plot_band_overlay(ax, primary_slice, selected=None, marker_scale=0.56)
    ax.set_title(title, fontsize=7.8, color=INK, pad=3.5)
    ax.set_xlabel(r"$\beta/\pi$", fontsize=6.4)
    ax.set_ylabel("")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_box_aspect(1)
    polish_phase_axis(ax, labelsize=5.8)
fig.subplots_adjust(left=0.047, right=0.940, bottom=0.19, top=0.91, wspace=0.12)
fig.canvas.draw()
right_box = delta_axes[-1].get_position()
cax = fig.add_axes([right_box.x1 + 0.010, right_box.y0, 0.012, right_box.height])
cbar = fig.colorbar(delta_im, cax=cax)
cbar.outline.set_visible(False)
cbar.set_ticks([0.0, 0.15, 0.30, 0.45])
cbar.ax.tick_params(labelsize=6.0, length=2.0, width=0.55, pad=1.6)
cbar.set_label("rank loss\n(dark = worse)", fontsize=6.2, labelpad=3.5)
savefig_dual(fig, "fig2_short_evidence")

# Optional QRC-only diagnostic figure retained as a repository artifact.
screen = pd.read_csv(DATA / "screening_retention_recomputed_intrinsic_diagnostics.csv")
diag = pd.read_csv(DATA / "qrc_real_current_intrinsic_diagnostics_with_perf.csv")
diag["logMC"] = np.log1p(diag["MC"])
spearman = pd.read_csv(DATA / "qrc_real_current_diagnostic_spearman_named.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.18))
dead_mc_threshold = 0.1
active_mc = diag[diag.MC > dead_mc_threshold]
axes[0].scatter(active_mc.logMC, active_mc.mean_val_rank, s=8.0, alpha=0.26, color=PHASE_ROSE, edgecolor="none")
z = np.polyfit(active_mc.logMC, active_mc.mean_val_rank, 1)
xs = np.linspace(active_mc.logMC.min(), active_mc.logMC.max(), 220)
axes[0].plot(xs, z[0] * xs + z[1], color="black", lw=1.0)
axes[0].set_title("(a) log memory", fontsize=8.8, pad=3)
axes[0].set_xlabel(r"$\log(1+\mathrm{MC})$", fontsize=7.8)
axes[0].set_ylabel("validation rank", fontsize=7.8)
sp_order = ["IPCtot", "MC", "IPCmem", "IPCnonlin", "Vfeat", "reff"]
sp_labels = [r"IPC$_t$", "MC", r"IPC$_m$", r"IPC$_n$", r"$V_f$", r"$r_e$"]
sp = spearman.set_index("metric").reindex(sp_order)
sp_y = np.arange(len(sp))[::-1]
sp_colors = [PHASE_GOLD, PHASE_ROSE, PHASE_CORAL, PHASE_AMBER, PHASE_VIOLET, PHASE_VIOLET]
axes[1].axvspan(-1.0, 0.0, color=PHASE_CORAL, alpha=0.055, lw=0)
axes[1].axvspan(0.0, 0.25, color=PHASE_VIOLET, alpha=0.075, lw=0)
axes[1].axvline(0, color=INK, lw=0.75, alpha=0.90)
for yi, label, val, color in zip(sp_y, sp_labels, sp.spearman_vs_val_rank.to_numpy(dtype=float), sp_colors):
    axes[1].hlines(yi, min(0.0, val), max(0.0, val), color=color, lw=2.0, alpha=0.68, zorder=2)
    axes[1].scatter([val], [yi], s=24, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    x_text = val + 0.055 if val < 0 else 0.045
    ha = "left" if val < 0 else "right"
    axes[1].text(
        x_text,
        yi,
        f"{val:.2f}",
        ha=ha,
        va="center",
        fontsize=5.8,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.66, pad=0.2),
    )
axes[1].set_yticks(sp_y, sp_labels)
axes[1].set_xlim(-1.0, 0.24)
axes[1].set_xticks([-1.0, -0.5, 0.0])
axes[1].set_ylim(-0.65, len(sp) - 0.35)
axes[1].set_title("(b) diagnostic rank", fontsize=8.8, pad=3)
axes[1].set_xlabel(r"Spearman $\rho_s$", fontsize=7.8)
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
    bbox_to_anchor=(1.045, 0.025),
    handlelength=1.6,
    borderaxespad=0.0,
)
for i, ax in enumerate(axes):
    if i == 1:
        ax.xaxis.grid(True, color=GRID, linewidth=0.40, alpha=0.55)
        ax.yaxis.grid(False)
    else:
        ax.grid(color=GRID, linewidth=0.42, alpha=0.68)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.8, length=2.0, width=0.55, pad=1.5)
fig.tight_layout(w_pad=0.95)
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
