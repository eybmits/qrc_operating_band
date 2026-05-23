#!/usr/bin/env python3
"""Task-wise validation-only QRC96 vs ESN100 analysis for the expanded same-architecture grid."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from analyze_qrc96_esn100 import bootstrap_ci, paired_tests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

QRC_PATH = DATA / "qrc96_same_arch_expanded_grid.csv"
QRC_METADATA_PATH = DATA / "qrc96_same_arch_expanded_metadata.json"
ESN_PATH = DATA / "esn_candidate_performance.csv"
STATS_PATH = DATA / "qrc96_esn100_taskwise_stats.json"
PER_TASK_PATH = DATA / "qrc96_esn100_taskwise_per_task.csv"
SEED_PAIRS_PATH = DATA / "qrc96_esn100_taskwise_seed_pairs.csv"
TASK_SEED_PAIRS_PATH = DATA / "qrc96_esn100_taskwise_task_seed_pairs.csv"
SELECTED_CONFIGS_PATH = DATA / "qrc96_esn100_taskwise_selected_configs.csv"
LATEX_OUTPUT_PATH = ROOT / "paper" / "generated" / "qrc96_taskwise_numbers.tex"

TASK_ORDER = ["mackey_glass", "lorenz", "narma10", "sunspots_annual"]
NON_FLOOR_TASKS = ["narma10", "sunspots_annual"]


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


def task_label(task: str) -> str:
    return {
        "mackey_glass": "Mackey--Glass",
        "lorenz": "Lorenz",
        "narma10": "NARMA10",
        "sunspots_annual": "Sunspots",
    }.get(task, task.replace("_", " "))


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


def selected_config_row(model: str, task: str, best: pd.Series) -> Dict[str, Any]:
    if model == "QRC96":
        return {
            "model": model,
            "task": task,
            "selection": "task-wise mean validation NMSE across seeds",
            "beta_pi": float(best["beta_pi"]),
            "lambda_pi": float(best["lambda_pi"]),
            "gamma": float(best["gamma"]),
            "features": 96,
            "mean_val_nmse": float(best["mean_val_nmse"]),
            "mean_holdout_nmse": float(best["mean_test_nmse"]),
            "rows": int(best["rows"]),
        }
    return {
        "model": model,
        "task": task,
        "selection": "task-wise mean validation NMSE across seeds",
        "units": int(best["units"]),
        "spectral_radius": float(best["sr"]),
        "input_scale": float(best["input_scale"]),
        "leak": float(best["leak"]),
        "features": int(best["units"]),
        "mean_val_nmse": float(best["mean_val_nmse"]),
        "mean_holdout_nmse": float(best["mean_test_nmse"]),
        "rows": int(best["rows"]),
    }


def gate_summary(per_task: pd.DataFrame, stats_by_task: Dict[str, Any]) -> Dict[str, Any]:
    narma = per_task[per_task["task"] == "narma10"].iloc[0].to_dict()
    sun = per_task[per_task["task"] == "sunspots_annual"].iloc[0].to_dict()
    sun_tests = stats_by_task["sunspots_annual"]
    narma_tests = stats_by_task["narma10"]
    sun_ci = sun_tests["delta"]["ci95_low"], sun_tests["delta"]["ci95_high"]
    return {
        "narma10_strong": bool(narma["delta_esn_minus_qrc"] > 0 and narma_tests["tests"]["wins"] >= 8),
        "sunspots_mean_lower": bool(sun["delta_esn_minus_qrc"] > 0),
        "sunspots_seed_gate": bool(sun_tests["tests"]["wins"] >= 7),
        "sunspots_ci_mostly_positive": bool(sun_ci[1] > 0 and sun_ci[0] > -0.01),
        "sunspots_full_gate": bool(
            sun["delta_esn_minus_qrc"] > 0
            and (sun_tests["tests"]["wins"] >= 7 or (sun_ci[1] > 0 and sun_ci[0] > -0.01))
        ),
        "non_floor_claim_allowed": bool(
            narma["delta_esn_minus_qrc"] > 0
            and narma_tests["tests"]["wins"] >= 8
            and sun["delta_esn_minus_qrc"] > 0
            and (sun_tests["tests"]["wins"] >= 7 or (sun_ci[1] > 0 and sun_ci[0] > -0.01))
        ),
    }


def write_latex_macros(stats: Dict[str, Any], per_task: pd.DataFrame) -> None:
    rows = []
    for task in TASK_ORDER:
        row = per_task[per_task["task"] == task].iloc[0]
        rows.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                task_label(task),
                fmt_table_float(row["esn100_mean_nmse"]),
                fmt_table_float(row["qrc96_mean_nmse"]),
                fmt_table_float(row["delta_esn_minus_qrc"]),
                row["qrc96_wins_over_seeds"],
                row["qrc96_selected_short"],
                row["esn100_selected_short"],
            )
        )

    non_floor = stats["non_floor_summary"]
    gates = stats["gates"]
    sun = stats["by_task"]["sunspots_annual"]
    narma = stats["by_task"]["narma10"]
    lines = [
        "% Auto-generated by scripts/analyze_qrc96_esn100_taskwise.py; do not edit by hand.",
        f"\\newcommand{{\\QRCTaskwiseNonFloorDeltaMean}}{{{fmt_float(non_floor['delta']['mean'], 4)}}}",
        f"\\newcommand{{\\QRCTaskwiseNonFloorCILow}}{{{fmt_float(non_floor['delta']['ci95_low'], 4)}}}",
        f"\\newcommand{{\\QRCTaskwiseNonFloorCIHigh}}{{{fmt_float(non_floor['delta']['ci95_high'], 4)}}}",
        f"\\newcommand{{\\QRCTaskwiseNonFloorWins}}{{{int(non_floor['tests']['wins'])}/{int(non_floor['tests']['n'])}}}",
        f"\\newcommand{{\\QRCTaskwiseNarmaDelta}}{{{fmt_float(narma['delta']['mean'], 4)}}}",
        f"\\newcommand{{\\QRCTaskwiseNarmaWins}}{{{int(narma['tests']['wins'])}/{int(narma['tests']['n'])}}}",
        f"\\newcommand{{\\QRCTaskwiseSunDelta}}{{{fmt_float(sun['delta']['mean'], 4)}}}",
        f"\\newcommand{{\\QRCTaskwiseSunCILow}}{{{fmt_float(sun['delta']['ci95_low'], 4)}}}",
        f"\\newcommand{{\\QRCTaskwiseSunCIHigh}}{{{fmt_float(sun['delta']['ci95_high'], 4)}}}",
        f"\\newcommand{{\\QRCTaskwiseSunWins}}{{{int(sun['tests']['wins'])}/{int(sun['tests']['n'])}}}",
        f"\\newcommand{{\\QRCTaskwiseSunWilcoxonGreaterP}}{{{fmt_p(sun['tests']['wilcoxon_greater_p'])}}}",
        f"\\newcommand{{\\QRCTaskwiseClaimAllowed}}{{{'yes' if gates['non_floor_claim_allowed'] else 'no'}}}",
        "\\newcommand{\\QRCTaskwiseRows}{%",
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
    esn100 = esn[esn["units"] == 100].copy()
    metadata = json.loads(QRC_METADATA_PATH.read_text()) if QRC_METADATA_PATH.exists() else {}

    pair_rows = []
    selected_configs = []
    per_task_rows = []
    by_task: Dict[str, Any] = {}

    for task in TASK_ORDER:
        qtask = qrc[qrc["task"] == task].copy()
        etask = esn100[esn100["task"] == task].copy()
        qbest, qsel = select_by_mean_validation(qtask, ["beta_pi", "lambda_pi", "gamma"])
        ebest, esel = select_by_mean_validation(etask, ["units", "sr", "input_scale", "leak"])
        selected_configs.append(selected_config_row("QRC96", task, qbest))
        selected_configs.append(selected_config_row("ESN100", task, ebest))

        q = qsel.rename(columns={"val_nmse": "qrc_val_nmse", "test_nmse": "qrc_test_nmse", "alpha": "qrc_alpha"})
        e = esel.rename(columns={"val_nmse": "esn_val_nmse", "test_nmse": "esn_test_nmse", "alpha": "esn_alpha"})
        pairs = q.merge(
            e[["task", "seed", "esn_val_nmse", "esn_test_nmse", "esn_alpha", "sr", "input_scale", "leak", "units"]],
            on=["task", "seed"],
            how="inner",
            validate="one_to_one",
        )
        if len(pairs) != 10:
            raise RuntimeError(f"Expected 10 task-wise seed pairs for {task}, got {len(pairs)}.")
        pairs["delta_esn_minus_qrc"] = pairs["esn_test_nmse"] - pairs["qrc_test_nmse"]
        pairs["qrc_wins"] = pairs["delta_esn_minus_qrc"] > 0
        pairs["qrc_selected_short"] = f"{float(qbest['beta_pi']):.3f}/{float(qbest['lambda_pi']):.3f}/{float(qbest['gamma']):.3f}"
        pairs["esn100_selected_short"] = f"{float(ebest['sr']):.1f}/{float(ebest['input_scale']):.1f}/{float(ebest['leak']):.1f}"
        pair_rows.append(pairs)

        delta = pairs["delta_esn_minus_qrc"].to_numpy(dtype=float)
        by_task[task] = {
            "delta": bootstrap_ci(delta),
            "tests": paired_tests(delta),
            "qrc_selected": selected_configs[-2],
            "esn100_selected": selected_configs[-1],
        }
        per_task_rows.append(
            {
                "task": task,
                "esn100_mean_nmse": float(pairs["esn_test_nmse"].mean()),
                "qrc96_mean_nmse": float(pairs["qrc_test_nmse"].mean()),
                "delta_esn_minus_qrc": float(delta.mean()),
                "qrc96_wins": int((delta > 0).sum()),
                "seeds": int(len(pairs)),
                "qrc96_wins_over_seeds": f"{int((delta > 0).sum())}/{len(pairs)}",
                "qrc96_selected_short": pairs["qrc_selected_short"].iloc[0],
                "esn100_selected_short": pairs["esn100_selected_short"].iloc[0],
            }
        )

    pair_df = pd.concat(pair_rows, ignore_index=True)
    per_task = pd.DataFrame(per_task_rows)
    selected_configs_df = pd.DataFrame(selected_configs)

    seed_pairs = (
        pair_df.groupby("seed", as_index=False)
        .agg(
            qrc_mean_nmse=("qrc_test_nmse", "mean"),
            esn_mean_nmse=("esn_test_nmse", "mean"),
            delta_esn_minus_qrc=("delta_esn_minus_qrc", "mean"),
            qrc_task_wins=("qrc_wins", "sum"),
        )
        .sort_values("seed")
    )

    non_floor_pairs = pair_df[pair_df["task"].isin(NON_FLOOR_TASKS)].copy()
    non_floor_delta = non_floor_pairs["delta_esn_minus_qrc"].to_numpy(dtype=float)
    stats = {
        "delta_definition": "Delta = holdout NMSE_ESN100 - holdout NMSE_QRC96; positive values favor QRC96.",
        "selection": "task-wise mean validation NMSE across seeds; no holdout values are used for selection.",
        "architecture_lock": metadata.get("architecture_lock", {}),
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
        "by_task": by_task,
        "seed_level_all_tasks": {
            "qrc96_mean_holdout_nmse": float(seed_pairs["qrc_mean_nmse"].mean()),
            "esn100_mean_holdout_nmse": float(seed_pairs["esn_mean_nmse"].mean()),
            "delta": bootstrap_ci(seed_pairs["delta_esn_minus_qrc"].to_numpy(dtype=float)),
            "tests": paired_tests(seed_pairs["delta_esn_minus_qrc"].to_numpy(dtype=float)),
        },
        "non_floor_tasks": NON_FLOOR_TASKS,
        "non_floor_summary": {
            "qrc96_mean_holdout_nmse": float(non_floor_pairs["qrc_test_nmse"].mean()),
            "esn100_mean_holdout_nmse": float(non_floor_pairs["esn_test_nmse"].mean()),
            "delta": bootstrap_ci(non_floor_delta),
            "tests": paired_tests(non_floor_delta),
        },
        "per_task": per_task.to_dict(orient="records"),
    }
    stats["gates"] = gate_summary(per_task, by_task)

    per_task.to_csv(PER_TASK_PATH, index=False)
    seed_pairs.to_csv(SEED_PAIRS_PATH, index=False)
    pair_df.to_csv(TASK_SEED_PAIRS_PATH, index=False)
    selected_configs_df.to_csv(SELECTED_CONFIGS_PATH, index=False)
    STATS_PATH.write_text(json.dumps(_jsonify(stats), indent=2))
    write_latex_macros(_jsonify(stats), per_task)
    print(json.dumps(_jsonify(stats), indent=2))


if __name__ == "__main__":
    main()
