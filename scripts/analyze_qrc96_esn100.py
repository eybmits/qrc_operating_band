#!/usr/bin/env python3
"""Shared-validation selection and paired statistics for QRC96 vs ESN100."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

QRC_PATH = DATA / "qrc96_same_arch_expanded_grid.csv"
QRC_METADATA_PATH = DATA / "qrc96_same_arch_expanded_metadata.json"
ESN_PATH = DATA / "esn_candidate_performance.csv"
STATS_PATH = DATA / "qrc96_esn100_stats.json"
PER_TASK_PATH = DATA / "qrc96_esn100_per_task.csv"
SEED_PAIRS_PATH = DATA / "qrc96_esn100_seed_pairs.csv"
TASK_SEED_PAIRS_PATH = DATA / "qrc96_esn100_task_seed_pairs.csv"
SELECTED_ROWS_PATH = DATA / "qrc96_esn100_selected_rows.csv"
SELECTED_CONFIGS_PATH = DATA / "qrc96_esn100_selected_configs.csv"
LATEX_OUTPUT_PATH = ROOT / "paper" / "generated" / "qrc96_esn100_numbers.tex"


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    return value


def select_by_mean_validation(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.Series, pd.DataFrame]:
    cols = list(cols)
    grouped = (
        df.groupby(cols, as_index=False)
        .agg(mean_val_nmse=("val_nmse", "mean"), mean_test_nmse=("test_nmse", "mean"), rows=("test_nmse", "size"))
        .sort_values(["mean_val_nmse", "mean_test_nmse"])
        .reset_index(drop=True)
    )
    best = grouped.iloc[0]
    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        if np.issubdtype(df[col].dtype, np.number):
            mask &= np.isclose(df[col].to_numpy(dtype=float), float(best[col]))
        else:
            mask &= df[col].astype(str).to_numpy() == str(best[col])
    return best, df[mask].copy()


def bootstrap_ci(values: np.ndarray, draws: int = 50000, seed: int = 20260523) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "draws": int(draws),
        "seed": int(seed),
    }


def paired_tests(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1e-15]
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    ties = int(np.sum(np.abs(values) <= 1e-15))
    out: Dict[str, Any] = {"n": int(len(values)), "wins": wins, "losses": losses, "ties": ties}
    if len(nonzero) == 0:
        out.update(
            {
                "wilcoxon_greater_p": None,
                "wilcoxon_two_sided_p": None,
                "sign_greater_p": None,
                "sign_two_sided_p": None,
            }
        )
        return out
    out["wilcoxon_greater_p"] = float(wilcoxon(nonzero, alternative="greater", zero_method="wilcox").pvalue)
    out["wilcoxon_two_sided_p"] = float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
    out["sign_greater_p"] = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
    out["sign_two_sided_p"] = float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue)
    return out


def load_qrc_metadata() -> Dict[str, Any]:
    if not QRC_METADATA_PATH.exists():
        return {}
    return json.loads(QRC_METADATA_PATH.read_text())


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_table_float(value: float) -> str:
    value = float(value)
    if 0 < abs(value) < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def fmt_p(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if value < 0.001:
        return r"\ensuremath{<0.001}"
    return f"{value:.4f}"


def task_label(task: str) -> str:
    return {
        "mackey_glass": "Mackey--Glass",
        "lorenz": "Lorenz",
        "narma10": "NARMA10",
        "sunspots_annual": "Sunspots",
    }.get(task, task.replace("_", " "))


def write_latex_macros(stats: Dict[str, Any], per_task: pd.DataFrame) -> None:
    seed = stats["seed_level"]
    task_seed = stats["task_seed_level"]
    qsel = stats["qrc96_selected"]
    esel = stats["esn100_selected"]

    rows = []
    for _, row in per_task.iterrows():
        rows.append(
            "{} & {} & {} & {} & {} \\\\".format(
                task_label(str(row["task"])),
                fmt_table_float(row["esn100_mean_nmse"]),
                fmt_table_float(row["qrc96_mean_nmse"]),
                fmt_table_float(row["delta_esn_minus_qrc"]),
                row["qrc96_wins_over_seeds"],
            )
        )
    rows.append(
        "All seed means & {} & {} & {} & {}/{} \\\\".format(
            fmt_float(seed["esn100_mean_holdout_nmse"], 4),
            fmt_float(seed["qrc96_mean_holdout_nmse"], 4),
            fmt_float(seed["delta"]["mean"], 4),
            int(seed["tests"]["wins"]),
            int(seed["tests"]["n"]),
        )
    )
    rows.append(
        "All task--seed pairs & {} & {} & {} & {}/{} \\\\".format(
            fmt_float(task_seed["esn100_mean_holdout_nmse"], 4),
            fmt_float(task_seed["qrc96_mean_holdout_nmse"], 4),
            fmt_float(task_seed["delta"]["mean"], 4),
            int(task_seed["tests"]["wins"]),
            int(task_seed["tests"]["n"]),
        )
    )

    lines = [
        "% Auto-generated by scripts/analyze_qrc96_esn100.py; do not edit by hand.",
        f"\\newcommand{{\\QRCNinetySixBetaPi}}{{{fmt_float(qsel['beta_pi'], 3)}}}",
        f"\\newcommand{{\\QRCNinetySixLambdaPi}}{{{fmt_float(qsel['lambda_pi'], 3)}}}",
        f"\\newcommand{{\\QRCNinetySixGamma}}{{{fmt_float(qsel['gamma'], 3)}}}",
        f"\\newcommand{{\\QRCNinetySixMeanVal}}{{{fmt_float(qsel['mean_val_nmse'], 3)}}}",
        f"\\newcommand{{\\QRCNinetySixMeanTest}}{{{fmt_float(qsel['mean_holdout_nmse'], 3)}}}",
        f"\\newcommand{{\\ESNHundredSpectralRadius}}{{{fmt_float(esel['spectral_radius'], 1)}}}",
        f"\\newcommand{{\\ESNHundredInputScale}}{{{fmt_float(esel['input_scale'], 1)}}}",
        f"\\newcommand{{\\ESNHundredLeak}}{{{fmt_float(esel['leak'], 1)}}}",
        f"\\newcommand{{\\ESNHundredMeanVal}}{{{fmt_float(esel['mean_val_nmse'], 3)}}}",
        f"\\newcommand{{\\ESNHundredMeanTest}}{{{fmt_float(esel['mean_holdout_nmse'], 3)}}}",
        f"\\newcommand{{\\QRCSeedDeltaMean}}{{{fmt_float(seed['delta']['mean'], 4)}}}",
        f"\\newcommand{{\\QRCSeedDeltaCILow}}{{{fmt_float(seed['delta']['ci95_low'], 4)}}}",
        f"\\newcommand{{\\QRCSeedDeltaCIHigh}}{{{fmt_float(seed['delta']['ci95_high'], 4)}}}",
        f"\\newcommand{{\\QRCSeedWins}}{{{int(seed['tests']['wins'])}/{int(seed['tests']['n'])}}}",
        f"\\newcommand{{\\QRCSeedWilcoxonGreaterP}}{{{fmt_p(seed['tests']['wilcoxon_greater_p'])}}}",
        f"\\newcommand{{\\QRCSeedSignGreaterP}}{{{fmt_p(seed['tests']['sign_greater_p'])}}}",
        f"\\newcommand{{\\QRCTaskSeedDeltaMean}}{{{fmt_float(task_seed['delta']['mean'], 4)}}}",
        f"\\newcommand{{\\QRCTaskSeedDeltaCILow}}{{{fmt_float(task_seed['delta']['ci95_low'], 4)}}}",
        f"\\newcommand{{\\QRCTaskSeedDeltaCIHigh}}{{{fmt_float(task_seed['delta']['ci95_high'], 4)}}}",
        f"\\newcommand{{\\QRCTaskSeedWins}}{{{int(task_seed['tests']['wins'])}/{int(task_seed['tests']['n'])}}}",
        f"\\newcommand{{\\QRCTaskSeedWilcoxonGreaterP}}{{{fmt_p(task_seed['tests']['wilcoxon_greater_p'])}}}",
        f"\\newcommand{{\\QRCTaskSeedSignGreaterP}}{{{fmt_p(task_seed['tests']['sign_greater_p'])}}}",
        "\\newcommand{\\QRCPerTaskStatsRows}{%",
        "\n".join(rows),
        "}",
    ]
    LATEX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEX_OUTPUT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not QRC_PATH.exists():
        raise SystemExit(f"Missing {QRC_PATH}; run scripts/run_qrc96_same_arch_expanded.py first.")
    qrc = pd.read_csv(QRC_PATH)
    esn = pd.read_csv(ESN_PATH)
    qmeta = load_qrc_metadata()

    qrc_best, qrc_sel = select_by_mean_validation(qrc, ["beta_pi", "lambda_pi", "gamma"])
    esn100 = esn[esn["units"] == 100].copy()
    esn_best, esn_sel = select_by_mean_validation(esn100, ["units", "sr", "input_scale", "leak"])

    q = qrc_sel.rename(columns={"val_nmse": "qrc_val_nmse", "test_nmse": "qrc_test_nmse", "alpha": "qrc_alpha"})
    e = esn_sel.rename(columns={"val_nmse": "esn_val_nmse", "test_nmse": "esn_test_nmse", "alpha": "esn_alpha"})
    pairs = q.merge(
        e[["task", "seed", "esn_val_nmse", "esn_test_nmse", "esn_alpha", "sr", "input_scale", "leak", "units"]],
        on=["task", "seed"],
        how="inner",
        validate="one_to_one",
    )
    expected_pairs = qrc_sel[["task", "seed"]].drop_duplicates().shape[0]
    if len(pairs) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} QRC/ESN pairs, got {len(pairs)}.")
    pairs["delta_esn_minus_qrc"] = pairs["esn_test_nmse"] - pairs["qrc_test_nmse"]
    pairs["qrc_wins"] = pairs["delta_esn_minus_qrc"] > 0

    seed_pairs = (
        pairs.groupby("seed", as_index=False)
        .agg(
            qrc_mean_nmse=("qrc_test_nmse", "mean"),
            esn_mean_nmse=("esn_test_nmse", "mean"),
            delta_esn_minus_qrc=("delta_esn_minus_qrc", "mean"),
            qrc_task_wins=("qrc_wins", "sum"),
        )
        .sort_values("seed")
    )
    per_task = (
        pairs.groupby("task", as_index=False)
        .agg(
            esn100_mean_nmse=("esn_test_nmse", "mean"),
            qrc96_mean_nmse=("qrc_test_nmse", "mean"),
            delta_esn_minus_qrc=("delta_esn_minus_qrc", "mean"),
            qrc96_wins=("qrc_wins", "sum"),
            seeds=("seed", "nunique"),
        )
        .sort_values("task")
    )
    per_task["qrc96_wins_over_seeds"] = per_task["qrc96_wins"].astype(int).astype(str) + "/" + per_task["seeds"].astype(int).astype(str)

    selected_rows = pd.concat(
        [
            qrc_sel.assign(model="QRC96"),
            esn_sel.assign(model="ESN100").rename(columns={"units": "features"}),
        ],
        ignore_index=True,
        sort=False,
    )
    selected_configs = pd.DataFrame(
        [
            {
                "model": "QRC96",
                "selection": "mean validation NMSE across all tasks and seeds",
                "beta_pi": float(qrc_best["beta_pi"]),
                "lambda_pi": float(qrc_best["lambda_pi"]),
                "gamma": float(qrc_best["gamma"]),
                "features": 96,
                "mean_val_nmse": float(qrc_best["mean_val_nmse"]),
                "mean_holdout_nmse": float(qrc_best["mean_test_nmse"]),
                "rows": int(qrc_best["rows"]),
            },
            {
                "model": "ESN100",
                "selection": "mean validation NMSE across all tasks and seeds",
                "units": int(esn_best["units"]),
                "spectral_radius": float(esn_best["sr"]),
                "input_scale": float(esn_best["input_scale"]),
                "leak": float(esn_best["leak"]),
                "features": int(esn_best["units"]),
                "mean_val_nmse": float(esn_best["mean_val_nmse"]),
                "mean_holdout_nmse": float(esn_best["mean_test_nmse"]),
                "rows": int(esn_best["rows"]),
            },
        ]
    )

    seed_delta = seed_pairs["delta_esn_minus_qrc"].to_numpy(dtype=float)
    pair_delta = pairs["delta_esn_minus_qrc"].to_numpy(dtype=float)
    stats = {
        "delta_definition": "Delta = holdout NMSE_ESN100 - holdout NMSE_QRC96; positive values favor QRC96.",
        "primary_unit": "seed-level task-mean holdout deltas, n=10",
        "secondary_unit": "task-seed holdout deltas, n=40",
        "qrc96_selected": selected_configs[selected_configs["model"] == "QRC96"].iloc[0].dropna().to_dict(),
        "esn100_selected": selected_configs[selected_configs["model"] == "ESN100"].iloc[0].dropna().to_dict(),
        "qrc96_grid": {
            "beta_pi": sorted(float(x) for x in qrc["beta_pi"].unique()),
            "lambda_pi": sorted(float(x) for x in qrc["lambda_pi"].unique()),
            "gamma": sorted(float(x) for x in qrc["gamma"].unique()),
            "rows": int(len(qrc)),
        },
        "esn100_grid": {
            "units": sorted(int(x) for x in esn100["units"].unique()),
            "spectral_radius": sorted(float(x) for x in esn100["sr"].unique()),
            "input_scale": sorted(float(x) for x in esn100["input_scale"].unique()),
            "leak": sorted(float(x) for x in esn100["leak"].unique()),
            "rows": int(len(esn100)),
        },
        "ridge_alpha_grid": qmeta.get("ridge_alpha_grid", [float(x) for x in np.logspace(-8, 3, 12)]),
        "seed_level": {
            "qrc96_mean_holdout_nmse": float(seed_pairs["qrc_mean_nmse"].mean()),
            "esn100_mean_holdout_nmse": float(seed_pairs["esn_mean_nmse"].mean()),
            "delta": bootstrap_ci(seed_delta),
            "tests": paired_tests(seed_delta),
        },
        "task_seed_level": {
            "qrc96_mean_holdout_nmse": float(pairs["qrc_test_nmse"].mean()),
            "esn100_mean_holdout_nmse": float(pairs["esn_test_nmse"].mean()),
            "delta": bootstrap_ci(pair_delta),
            "tests": paired_tests(pair_delta),
        },
        "per_task": per_task.to_dict(orient="records"),
    }

    per_task.to_csv(PER_TASK_PATH, index=False)
    seed_pairs.to_csv(SEED_PAIRS_PATH, index=False)
    pairs.to_csv(TASK_SEED_PAIRS_PATH, index=False)
    selected_rows.to_csv(SELECTED_ROWS_PATH, index=False)
    selected_configs.to_csv(SELECTED_CONFIGS_PATH, index=False)
    STATS_PATH.write_text(json.dumps(_jsonify(stats), indent=2))
    write_latex_macros(_jsonify(stats), per_task)
    print(json.dumps(_jsonify(stats), indent=2))


if __name__ == "__main__":
    main()
