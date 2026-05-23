#!/usr/bin/env python3
"""Analyze QRC-only phase-map transfer, bands, and ablation retention."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GENERATED = ROOT / "paper" / "generated"
GENERATED.mkdir(parents=True, exist_ok=True)

KEYS = ["beta_pi", "lambda_pi", "gamma"]
PS = [10, 15, 20, 25, 30]
QS = [0.5, 0.6, 0.7, 0.8]
PRIMARY_P = 20
PRIMARY_Q = 0.7
BOOT_DRAWS = 50000
BOOT_SEED = 20260523


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def json_clean(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def bootstrap_ci(values: Sequence[float], draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(draws, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "draws": int(draws),
        "seed": int(seed),
        "n": int(len(arr)),
    }


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["replicate"] = d["task"] + "__seed" + d["seed"].astype(str)
    d["val_rank_pct"] = d.groupby("replicate")["val_nmse"].rank(method="average", pct=True)
    d["test_rank_pct"] = d.groupby("replicate")["test_nmse"].rank(method="average", pct=True)
    return d


def band_table(df: pd.DataFrame, p: int, q: float, replicate_values: Iterable[str] | None = None) -> pd.DataFrame:
    d = df if replicate_values is None else df[df["replicate"].isin(set(replicate_values))]
    h = d.copy()
    h["top"] = h["val_rank_pct"] <= p / 100.0
    agg = (
        h.groupby(KEYS)
        .agg(top_frequency=("top", "mean"), mean_val_rank=("val_rank_pct", "mean"), mean_val_nmse=("val_nmse", "mean"))
        .reset_index()
    )
    agg["in_band"] = agg["top_frequency"] >= q
    return agg


def connected_components(points: pd.DataFrame, all_points: pd.DataFrame) -> Dict[str, int]:
    if points.empty:
        return {"components": 0, "largest_component": 0}
    grids = {k: sorted(all_points[k].unique()) for k in KEYS}
    idx_maps = {k: {float(v): i for i, v in enumerate(vals)} for k, vals in grids.items()}
    point_indices = {
        tuple(idx_maps[k][float(row[k])] for k in KEYS)
        for _, row in points[KEYS].drop_duplicates().iterrows()
    }
    seen = set()
    sizes: List[int] = []
    for start in point_indices:
        if start in seen:
            continue
        seen.add(start)
        q = deque([start])
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for axis in range(3):
                for step in (-1, 1):
                    nxt = list(cur)
                    nxt[axis] += step
                    nxt_t = tuple(nxt)
                    if nxt_t in point_indices and nxt_t not in seen:
                        seen.add(nxt_t)
                        q.append(nxt_t)
        sizes.append(size)
    return {"components": int(len(sizes)), "largest_component": int(max(sizes))}


def medoid_point(points: pd.DataFrame) -> Dict[str, float]:
    arr = points[KEYS].drop_duplicates().to_numpy(dtype=float)
    if len(arr) == 0:
        return {k: float("nan") for k in KEYS}
    z = (arr - arr.mean(axis=0)) / np.where(arr.std(axis=0) == 0, 1.0, arr.std(axis=0))
    dist = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    row = arr[int(np.argmin(dist.sum(axis=1)))]
    return {k: float(v) for k, v in zip(KEYS, row)}


def key_tuples(df: pd.DataFrame, keys: Sequence[str] = KEYS) -> set[Tuple[float, ...]]:
    return {tuple(float(row[k]) for k in keys) for _, row in df[list(keys)].iterrows()}


def mean_rank_for_keys(df: pd.DataFrame, keys: set[Tuple[float, ...]], by: List[str]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame(columns=by + ["mean_val_rank"])
    d = df[df[KEYS].apply(lambda r: tuple(float(r[k]) for k in KEYS) in keys, axis=1)]
    return d.groupby(by).agg(mean_val_rank=("val_rank_pct", "mean")).reset_index()


def main() -> None:
    q = add_ranks(pd.read_csv(DATA / "qrc_seed_ensemble_grid.csv"))
    all_points = q[KEYS].drop_duplicates().reset_index(drop=True)
    band_rows: List[pd.DataFrame] = []
    band_summaries: Dict[str, Dict[str, object]] = {}

    for p in PS:
        for qthr in QS:
            bt = band_table(q, p, qthr)
            comps = connected_components(bt[bt["in_band"]], all_points)
            band = bt[bt["in_band"]]
            medoid = medoid_point(band)
            gamma_counts = {str(k): int(v) for k, v in band["gamma"].value_counts().sort_index().items()}
            key = f"B{p}_q{str(qthr).replace('.', 'p')}"
            band_summaries[key] = {
                "p": p,
                "q": qthr,
                "size": int(len(band)),
                **comps,
                "gamma_counts": gamma_counts,
                "medoid": medoid,
                "mean_val_rank": float(band["mean_val_rank"].mean()) if len(band) else None,
            }
            bt = bt.copy()
            bt["p"] = p
            bt["q"] = qthr
            band_rows.append(bt)

    membership = pd.concat(band_rows, ignore_index=True)
    membership[["p", "q", *KEYS, "top_frequency", "mean_val_rank", "mean_val_nmse", "in_band"]].to_csv(
        DATA / "phase_map_band_membership.csv", index=False
    )

    primary = membership[(membership["p"] == PRIMARY_P) & np.isclose(membership["q"], PRIMARY_Q)]
    primary_keys = key_tuples(primary[primary["in_band"]])

    loto_values: List[float] = []
    loto_rows: List[Dict[str, object]] = []
    tasks = sorted(q["task"].unique())
    for task in tasks:
        train_reps = q.loc[q["task"] != task, "replicate"].unique()
        bt = band_table(q, PRIMARY_P, PRIMARY_Q, train_reps)
        keys = key_tuples(bt[bt["in_band"]])
        held = mean_rank_for_keys(q[q["task"] == task], keys, ["task", "seed"])
        loto_values.extend(held["mean_val_rank"].tolist())
        loto_rows.append(
            {
                "heldout_task": task,
                "band_size": int(len(keys)),
                "mean_heldout_rank": float(held["mean_val_rank"].mean()),
                "median_heldout_rank": float(held["mean_val_rank"].median()),
                **{f"rep_{k}": v for k, v in medoid_point(bt[bt["in_band"]]).items()},
            }
        )
    loto = pd.DataFrame(loto_rows)
    loto.to_csv(DATA / "phase_map_leave_one_task_out.csv", index=False)

    loso_values: List[float] = []
    loso_rows: List[Dict[str, object]] = []
    for seed in sorted(q["seed"].unique()):
        train_reps = q.loc[q["seed"] != seed, "replicate"].unique()
        bt = band_table(q, PRIMARY_P, PRIMARY_Q, train_reps)
        keys = key_tuples(bt[bt["in_band"]])
        held = mean_rank_for_keys(q[q["seed"] == seed], keys, ["seed", "task"])
        loso_values.extend(held["mean_val_rank"].tolist())
        loso_rows.append(
            {
                "heldout_seed": int(seed),
                "band_size": int(len(keys)),
                "mean_heldout_rank": float(held["mean_val_rank"].mean()),
                "median_heldout_rank": float(held["mean_val_rank"].median()),
                **{f"rep_{k}": v for k, v in medoid_point(bt[bt["in_band"]]).items()},
            }
        )
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(DATA / "phase_map_leave_one_seed_out.csv", index=False)

    transfer_matrix = mean_rank_for_keys(q, primary_keys, ["task", "seed"])
    transfer_matrix.to_csv(DATA / "phase_map_task_seed_transfer_matrix.csv", index=False)

    ablation_path = DATA / "qrc_phase_ablation_slice_grid.csv"
    if not ablation_path.exists():
        raise FileNotFoundError(f"Missing {ablation_path}; run scripts/run_qrc_phase_ablation_slices.py first.")
    abl = pd.read_csv(ablation_path)
    abl["replicate"] = abl["variant"] + "__" + abl["task"] + "__seed" + abl["seed"].astype(str)
    abl["val_rank_pct"] = abl.groupby("replicate")["val_nmse"].rank(method="average", pct=True)
    abl["top20"] = abl["val_rank_pct"] <= PRIMARY_P / 100.0
    slice_keys = ["beta_pi", "lambda_pi"]
    freq = (
        abl.groupby(["variant", *slice_keys])
        .agg(top20_frequency=("top20", "mean"), mean_val_rank=("val_rank_pct", "mean"), mean_val_nmse=("val_nmse", "mean"))
        .reset_index()
    )
    # Retention is defined against the primary 3D base band, projected to the
    # central gamma slice used by the ablation maps. Re-ranking only inside the
    # 2D slice would make the base band empty for this coarse grid.
    base_slice_keys = {
        (b, l)
        for b, l, g in primary_keys
        if np.isclose(g, 0.12)
    }
    ablation_rows: List[Dict[str, object]] = []
    for variant, g in freq.groupby("variant"):
        if variant == "base_amplitude_rxzz":
            variant_band = set(base_slice_keys)
        else:
            variant_band = key_tuples(g[g["top20_frequency"] >= PRIMARY_Q], slice_keys)
        retained = len(base_slice_keys & variant_band)
        on_base = g[g[slice_keys].apply(lambda r: tuple(float(r[k]) for k in slice_keys) in base_slice_keys, axis=1)]
        ablation_rows.append(
            {
                "variant": variant,
                "band_size": int(len(variant_band)),
                "base_band_size": int(len(base_slice_keys)),
                "retained_base_band_points": int(retained),
                "retention_fraction": float(retained / len(base_slice_keys)) if base_slice_keys else float("nan"),
                "mean_top20_frequency_on_base_band": float(on_base["top20_frequency"].mean()) if len(on_base) else float("nan"),
                "mean_rank_on_base_band": float(on_base["mean_val_rank"].mean()) if len(on_base) else float("nan"),
            }
        )
    ablation = pd.DataFrame(ablation_rows).sort_values("mean_rank_on_base_band")
    ablation.to_csv(DATA / "phase_map_ablation_retention.csv", index=False)

    spearman = pd.read_csv(DATA / "qrc_real_current_diagnostic_spearman_named.csv")
    sp = dict(zip(spearman["metric"], spearman["spearman_vs_val_rank"]))
    loto_ci = bootstrap_ci(loto_values)
    loso_ci = bootstrap_ci(loso_values)
    primary_key = f"B{PRIMARY_P}_q{str(PRIMARY_Q).replace('.', 'p')}"
    primary_summary = band_summaries[primary_key]

    stats = {
        "definition": "B_{p,q} = points whose validation rank is in the top p percent for at least q of task-seed replicates.",
        "primary_band": primary_key,
        "bands": band_summaries,
        "leave_one_task_out": {**loto_ci, "rows": loto_rows},
        "leave_one_seed_out": {**loso_ci, "rows": loso_rows},
        "ablation_retention": ablation.to_dict(orient="records"),
        "diagnostic_spearman": {k: float(v) for k, v in sp.items()},
    }
    stats = json_clean(stats)
    (DATA / "phase_map_generalization_stats.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")

    labels = {
        "base_amplitude_rxzz": "base AD+RxZZ",
        "gamma0_amplitude": r"$\gamma=0$",
        "dephasing": "dephasing",
        "depolarizing": "depolarizing",
        "mixer_none": "no mixer",
        "mixer_rx_only": "Rx only",
        "mixer_zz_only": "ZZ only",
    }
    order = ["base_amplitude_rxzz", "gamma0_amplitude", "dephasing", "depolarizing", "mixer_none", "mixer_rx_only", "mixer_zz_only"]
    abl_index = ablation.set_index("variant")
    rows = []
    for variant in order:
        row = abl_index.loc[variant]
        rows.append(
            rf"{labels[variant]} & {int(row.band_size)} & {fmt(row.retention_fraction, 2)} & {fmt(row.mean_rank_on_base_band)}\\"
        )

    macros = [
        "% Auto-generated by scripts/analyze_phase_map_generalization.py",
        macro("PhasePrimaryBand", rf"$B_{{{PRIMARY_P},0.7}}$"),
        macro("PhaseBTwentyQSeventySize", int(primary_summary["size"])),
        macro("PhaseBTwentyQSeventyComponents", int(primary_summary["components"])),
        macro("PhaseBTwentyQSeventyLargest", int(primary_summary["largest_component"])),
        macro("PhaseBTwentyQSixtySize", int(band_summaries["B20_q0p6"]["size"])),
        macro("PhaseBThirtyQSeventySize", int(band_summaries["B30_q0p7"]["size"])),
        macro("PhaseRepresentativeBetaPi", fmt(primary_summary["medoid"]["beta_pi"], 2)),
        macro("PhaseRepresentativeLambdaPi", fmt(primary_summary["medoid"]["lambda_pi"], 2)),
        macro("PhaseRepresentativeGamma", fmt(primary_summary["medoid"]["gamma"], 2)),
        macro("PhaseLotoMeanRank", fmt(loto_ci["mean"])),
        macro("PhaseLotoCILow", fmt(loto_ci["ci95_low"])),
        macro("PhaseLotoCIHigh", fmt(loto_ci["ci95_high"])),
        macro("PhaseLosoMeanRank", fmt(loso_ci["mean"])),
        macro("PhaseLosoCILow", fmt(loso_ci["ci95_low"])),
        macro("PhaseLosoCIHigh", fmt(loso_ci["ci95_high"])),
        macro("PhaseMCSpearman", fmt(sp["MC"], 2)),
        macro("PhaseIPCmemSpearman", fmt(sp["IPCmem"], 2)),
        macro("PhaseIPCtotSpearman", fmt(sp["IPCtot"], 2)),
        macro("PhaseIPCnonlinSpearman", fmt(sp["IPCnonlin"], 2)),
        macro("PhaseVfeatSpearman", fmt(sp["Vfeat"], 2)),
        macro("PhaseReffSpearman", fmt(sp["reff"], 2)),
        macro("PhaseAblationBaseBandSize", int(abl_index.loc["base_amplitude_rxzz", "base_band_size"])),
        macro("PhaseGammaZeroRetention", fmt(abl_index.loc["gamma0_amplitude", "retention_fraction"], 2)),
        macro("PhaseNoMixerRetention", fmt(abl_index.loc["mixer_none", "retention_fraction"], 2)),
        macro("PhaseDephaseRetention", fmt(abl_index.loc["dephasing", "retention_fraction"], 2)),
        macro("PhaseAblationRows", "\n".join(rows)),
        "",
    ]
    (GENERATED / "phase_map_numbers.tex").write_text("\n".join(macros))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
