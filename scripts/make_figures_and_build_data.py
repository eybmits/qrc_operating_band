#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
PHASE_GOLD = PHASE_CMAP(0.08)
PHASE_AMBER = PHASE_CMAP(0.18)
PHASE_CORAL = PHASE_CMAP(0.32)
PHASE_ROSE = PHASE_CMAP(0.46)
PHASE_PLUM = PHASE_CMAP(0.66)
PHASE_VIOLET = PHASE_CMAP(0.82)
PHASE_DARK = PHASE_CMAP(0.94)
INK = "#1f2933"
GRID = "#d9dee7"
SOFT_GRID = "#eef1f5"

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


def plot_rank_map(ax, d, title, xmax, ymax, selected=None, core=None, labelsize=6.5):
    agg = d.groupby(["beta_pi", "lambda_pi"]).agg(mean_rank=("val_rank_pct", "mean")).reset_index()
    XI, YI, Zs = smooth_grid(agg.beta_pi.values, agg.lambda_pi.values, agg.mean_rank.values, xmax, ymax)
    im = ax.imshow(
        Zs,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap="magma_r",
        vmin=0.05,
        vmax=0.8,
        aspect="auto",
    )
    ax.contour(XI, YI, Zs, levels=[0.15, 0.25, 0.40], colors="white", linewidths=[0.55, 0.45, 0.35], alpha=0.78)
    if core is not None and len(core):
        ax.scatter(core.beta_pi, core.lambda_pi, s=14, color=PHASE_GOLD, edgecolor=INK, linewidth=0.25, zorder=4)
    if selected is not None:
        ax.scatter([selected["beta_pi"]], [selected["lambda_pi"]], marker="*", s=44, color=PHASE_GOLD, edgecolor="black", linewidth=0.35, zorder=5)
    ax.set_title(title, fontsize=labelsize + 0.8, pad=2.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=labelsize, length=2)
    return im


def core_points_at_slice(d, p=0.20, q=0.70):
    chunks = []
    for _, g in d.groupby("replicate"):
        h = g.copy()
        h["top"] = h.val_rank_pct <= p
        chunks.append(h)
    return pd.concat(chunks).groupby(["beta_pi", "lambda_pi"]).top.mean().reset_index().query("top >= @q")


def variant_label(v: str) -> str:
    return {
        "base_amplitude_rxzz": "base",
        "gamma0_amplitude": r"$\gamma=0$",
        "dephasing": "dephase",
        "depolarizing": "depol.",
        "mixer_none": "no mix",
        "mixer_rx_only": "Rx only",
        "mixer_zz_only": "ZZ only",
    }.get(v, v)


def phase_luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


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

# Figure 1: full-size and compact leave-one-task-out phase maps.
panels = [("all tasks", all_tasks)] + [
    (f"leave out {TASK_LABELS[t]}", [x for x in all_tasks if x != t])
    for t in ["mackey_glass", "narma10", "lorenz", "sunspots_annual"]
]
fig, axes = plt.subplots(1, 5, figsize=(13.8, 3.35), sharex=True, sharey=True)
for ax, (title, tasks) in zip(axes, panels):
    d = qg[qg.task.isin(tasks)].copy()
    core = core_points_at_slice(d)
    im = plot_rank_map(ax, d, title, xmax, ymax, selected=selected, core=core, labelsize=8)
    ax.set_xlabel(r"$\beta/\pi$", fontsize=10)
axes[0].set_ylabel(r"$\lambda/\pi$", fontsize=11)
fig.suptitle(r"Memory-defined QRC operating regime at the $\gamma=0.12$ slice", fontsize=15, y=1.05)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.018)
cbar.set_label("validation rank percentile", fontsize=9)
cbar.ax.tick_params(labelsize=8)
savefig_dual(fig, "fig1_operating_regime_fixed_gamma")

fig, axes = plt.subplots(1, 5, figsize=(7.25, 1.72), sharex=True, sharey=True)
for ax, (title, tasks) in zip(axes, panels):
    d = qg[qg.task.isin(tasks)].copy()
    core = core_points_at_slice(d)
    im = plot_rank_map(ax, d, title, xmax, ymax, selected=selected, core=core, labelsize=6)
    ax.set_xlabel(r"$\beta/\pi$", fontsize=7)
axes[0].set_ylabel(r"$\lambda/\pi$", fontsize=7.5)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.012)
cbar.set_label("validation rank percentile", fontsize=7)
cbar.ax.tick_params(labelsize=6, length=2)
fig.suptitle(r"Phase maps transfer under leave-one-task-out conditioning ($\gamma=0.12$)", fontsize=9.5, y=1.08, color=INK)
savefig_dual(fig, "fig1_short_phase_maps")

# Figure 2: QRC-only atlas, ablations, transfer, diagnostics.
abl = pd.read_csv(DATA / "qrc_phase_ablation_slice_grid.csv")
abl["replicate"] = abl["variant"] + "__" + abl["task"] + "__seed" + abl["seed"].astype(str)
abl["val_rank_pct"] = abl.groupby("replicate")["val_nmse"].rank(method="average", pct=True)
fig = plt.figure(figsize=(7.25, 4.95))
gs = fig.add_gridspec(2, 2, hspace=0.54, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.set_title("(a) gamma atlas, all tasks", fontsize=8.2, color=INK, pad=4)
for i, gamma in enumerate([0.0, 0.05, 0.12, 0.22, 0.30]):
    iax = ax.inset_axes([0.01 + i * 0.195, 0.07, 0.18, 0.78])
    d = q[np.isclose(q.gamma, gamma)]
    core = primary_slice if np.isclose(gamma, gamma_star) else None
    plot_rank_map(iax, d, rf"$\gamma={gamma:.2f}$", xmax, ymax, selected=selected if np.isclose(gamma, gamma_star) else None, core=core, labelsize=4.9)
    iax.set_xticks([])
    iax.set_yticks([])
ax.text(0.5, 0.00, r"$\beta/\pi$  (horizontal), $\lambda/\pi$  (vertical)", transform=ax.transAxes, ha="center", fontsize=6.2, color=INK)

ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_title("(b) mechanism ablation slices", fontsize=8.2, color=INK, pad=4)
ablation_variants = ["base_amplitude_rxzz", "gamma0_amplitude", "mixer_none", "dephasing"]
for i, variant in enumerate(ablation_variants):
    row, col = divmod(i, 2)
    iax = ax.inset_axes([0.03 + col * 0.49, 0.52 - row * 0.49, 0.43, 0.39])
    d = abl[abl.variant == variant]
    agg = d.groupby(["beta_pi", "lambda_pi"]).agg(mean_rank=("val_rank_pct", "mean")).reset_index()
    XI, YI, Zs = smooth_grid(agg.beta_pi.values, agg.lambda_pi.values, agg.mean_rank.values, xmax, ymax, nx=160, ny=130)
    iax.imshow(Zs, origin="lower", extent=[0, xmax, 0, ymax], cmap="magma_r", vmin=0.05, vmax=0.8, aspect="auto")
    iax.contour(XI, YI, Zs, levels=[0.20, 0.35], colors="white", linewidths=[0.4, 0.3], alpha=0.72)
    iax.scatter(primary_slice.beta_pi, primary_slice.lambda_pi, s=12, facecolor="none", edgecolor=INK, linewidth=0.45, zorder=4)
    iax.set_title(variant_label(variant), fontsize=6.5, pad=1.5)
    iax.set_xlim(0, xmax)
    iax.set_ylim(0, ymax)
    iax.set_xticks([])
    iax.set_yticks([])
    iax.tick_params(labelsize=5, length=1.5)

ax = fig.add_subplot(gs[1, 0])
transfer = pd.read_csv(DATA / "phase_map_task_seed_transfer_matrix.csv")
task_order = ["mackey_glass", "lorenz", "narma10", "sunspots_annual"]
mat = transfer.pivot(index="task", columns="seed", values="mean_val_rank").reindex(task_order)
norm = plt.Normalize(vmin=0.04, vmax=0.54)
im = ax.imshow(mat.values, cmap=PHASE_CMAP, norm=norm, aspect="auto", interpolation="nearest")
ax.set_title("(c) primary-band task/seed transfer", fontsize=8.2, color=INK, pad=4)
ax.set_yticks(np.arange(len(task_order)), [TASK_LABELS[t] for t in task_order])
ax.set_xticks(np.arange(len(mat.columns)), [str(int(s)) for s in mat.columns])
ax.tick_params(axis="x", labelsize=5.7, length=0, pad=1.5, top=True, labeltop=True, bottom=False, labelbottom=False)
ax.tick_params(axis="y", labelsize=6.2, length=0, pad=2.5)
ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.72, alpha=0.78)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat.values[i, j]
        rgba = PHASE_CMAP(norm(val))
        txt_color = INK if phase_luminance(rgba) > 0.48 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.45, color=txt_color, alpha=0.92)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=5, length=2, width=0.55)
cbar.set_label("rank (lower better)", fontsize=5.8)

ax = fig.add_subplot(gs[1, 1])
screen = pd.read_csv(DATA / "screening_retention_recomputed_intrinsic_diagnostics.csv")
curve_specs = [
    ("IPCtot", r"IPC$_{\mathrm{tot}}$", PHASE_GOLD, 1.9, "-"),
    ("MC", "MC", PHASE_ROSE, 1.7, "-"),
    ("IPCmem", r"IPC$_{\mathrm{mem}}$", PHASE_CORAL, 1.7, "-"),
    ("Vfeat", r"$V_{\mathrm{feat}}$", PHASE_VIOLET, 1.55, "-"),
    ("random", "random", "#a8b0bd", 1.25, "--"),
]
x_dense = np.linspace(float(screen.budget_pct.min()), float(screen.budget_pct.max()), 260)
for col, _, color, lw, ls in curve_specs:
    y_dense = PchipInterpolator(screen["budget_pct"], screen[col])(x_dense)
    ax.plot(x_dense, y_dense, color=color, lw=lw, ls=ls, solid_capstyle="round", alpha=0.98)
    if col != "random":
        ax.scatter(screen["budget_pct"], screen[col], s=5.5, color=color, edgecolor="white", linewidth=0.22, zorder=3)
ax.set_title("(d) diagnostics recover the regime", fontsize=8.2, color=INK, pad=4)
ax.set_xlabel("screening budget (%)", fontsize=7)
ax.set_ylabel("top-decile retained (%)", fontsize=7)
ax.set_xlim(5, 108)
ax.set_ylim(-2, 104)
ax.set_xticks([20, 40, 60, 80, 100])
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(color=SOFT_GRID, linewidth=0.52, alpha=0.95)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.spines["left"].set_color("#27313d")
ax.spines["bottom"].set_color("#27313d")
ax.tick_params(labelsize=6)
inline_labels = {
    "IPCtot": (46, 99),
    "MC": (38, 93),
    "IPCmem": (30, 78),
    "Vfeat": (78, 93),
    "random": (83, 80),
}
for col, label, color, _, _ in curve_specs:
    x, y = inline_labels[col]
    ax.text(x, y, label, color=color, fontsize=5.9, ha="left", va="center", clip_on=False)
fig.suptitle("QRC-only evidence: phase atlas, mechanism stress tests, and diagnostics", fontsize=9.0, y=1.01, color=INK)
savefig_dual(fig, "fig2_short_evidence", aliases=("fig2_phase_map_ablation_evidence",))

# Optional QRC-only diagnostic figure retained as a repository artifact.
diag = pd.read_csv(DATA / "qrc_real_current_intrinsic_diagnostics_with_perf.csv")
diag["logMC"] = np.log1p(diag["MC"])
spearman = pd.read_csv(DATA / "qrc_real_current_diagnostic_spearman_named.csv")
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.1))
nz = diag[diag.MC > 0]
zero = diag[diag.MC == 0]
axes[0].scatter(nz.MC, nz.mean_val_rank, s=8, alpha=0.25, color=PHASE_ROSE, edgecolor="none", label="MC>0")
axes[0].scatter(zero.MC, zero.mean_val_rank, s=9, alpha=0.52, color=PHASE_AMBER, edgecolor="none", label="MC=0")
if len(nz):
    z = np.polyfit(nz.MC, nz.mean_val_rank, 1)
    xs = np.linspace(nz.MC.min(), nz.MC.max(), 200)
    axes[0].plot(xs, z[0] * xs + z[1], color="black", lw=1.0)
axes[0].set_title("(a) memory capacity")
axes[0].set_xlabel("MC")
axes[0].set_ylabel("validation rank")
axes[0].legend(frameon=False, fontsize=7)
axes[1].scatter(diag.logMC, diag.mean_val_rank, s=8, alpha=0.28, color=PHASE_ROSE, edgecolor="none")
z = np.polyfit(diag.logMC, diag.mean_val_rank, 1)
xs = np.linspace(diag.logMC.min(), diag.logMC.max(), 200)
axes[1].plot(xs, z[0] * xs + z[1], color="black", lw=1.0)
axes[1].set_title("(b) log memory")
axes[1].set_xlabel(r"$\log(1+\mathrm{MC})$")
axes[1].set_ylabel("validation rank")
sp_order = ["MC", "IPCmem", "IPCtot", "IPCnonlin", "Vfeat", "reff"]
sp = spearman.set_index("metric").reindex(sp_order)
axes[2].barh(np.arange(len(sp)), sp.spearman_vs_val_rank, color=[PHASE_CORAL if v < 0 else PHASE_VIOLET for v in sp.spearman_vs_val_rank])
axes[2].axvline(0, color="black", lw=0.8)
axes[2].set_yticks(np.arange(len(sp)), ["MC", r"IPC$_m$", r"IPC$_t$", r"IPC$_n$", r"$V_f$", r"$r_e$"])
axes[2].invert_yaxis()
axes[2].set_title("(c) diagnostic rank correlation")
axes[2].set_xlabel(r"Spearman $\rho_s$")
for col, color in [("IPCtot", PHASE_GOLD), ("MC", PHASE_ROSE), ("Vfeat", PHASE_VIOLET), ("random", "#9ca3af")]:
    axes[3].plot(screen["budget_pct"], screen[col], label=col, color=color, lw=1.4)
axes[3].set_title("(d) screening retention")
axes[3].set_xlabel("budget (%)")
axes[3].set_ylabel("retained (%)")
axes[3].legend(frameon=False, fontsize=7)
for ax in axes:
    ax.grid(color=GRID, linewidth=0.45, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
fig.tight_layout(w_pad=1.5)
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
