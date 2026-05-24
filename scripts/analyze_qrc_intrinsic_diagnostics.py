#!/usr/bin/env python3
"""Build QRC-only intrinsic diagnostic summaries used by the paper figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DIAGNOSTIC_INPUT = DATA / "qrc_real_current_intrinsic_diagnostics_with_perf.csv"
SPEARMAN_OUTPUT = DATA / "qrc_real_current_diagnostic_spearman_named.csv"
RETENTION_OUTPUT = DATA / "screening_retention_recomputed_intrinsic_diagnostics.csv"
SUMMARY_OUTPUT = DATA / "qrc_intrinsic_diagnostic_summary.json"

METRICS = ["MC", "IPCmem", "IPCtot", "IPCnonlin", "Vfeat", "reff"]
BUDGETS = np.arange(5, 101, 5)


def retention(df: pd.DataFrame, score: str, ascending: bool = False) -> np.ndarray:
    positives = max(1, int(df["is_top10"].sum()))
    vals = []
    for budget in BUDGETS:
        keep = max(1, int(np.ceil(len(df) * budget / 100.0)))
        kept = df.sort_values(score, ascending=ascending).head(keep)
        vals.append(100.0 * kept["is_top10"].sum() / positives)
    return np.asarray(vals, dtype=float)


def main() -> None:
    if not DIAGNOSTIC_INPUT.exists():
        raise FileNotFoundError(
            f"Missing {DIAGNOSTIC_INPUT}. Run scripts/compute_qrc_real_diagnostics.py "
            "or restore the checked-in diagnostic artifact first."
        )

    diag = pd.read_csv(DIAGNOSTIC_INPUT)
    required = {"mean_val_rank", *METRICS}
    missing = sorted(required - set(diag.columns))
    if missing:
        raise ValueError(f"{DIAGNOSTIC_INPUT} is missing required columns: {missing}")

    diag = diag.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    diag["is_top10"] = diag["mean_val_rank"] <= diag["mean_val_rank"].quantile(0.10)

    curves = {
        "IPCmem": retention(diag, "IPCmem", ascending=False),
        "MC": retention(diag, "MC", ascending=False),
        "IPCtot": retention(diag, "IPCtot", ascending=False),
        "Vfeat": retention(diag, "Vfeat", ascending=False),
        "reff": retention(diag, "reff", ascending=False),
        "random": BUDGETS.astype(float),
    }
    curve_df = pd.DataFrame({"budget_pct": BUDGETS})
    for key, values in curves.items():
        curve_df[key] = values
    curve_df.to_csv(RETENTION_OUTPUT, index=False)

    corr_rows = []
    for metric in METRICS:
        rho = float(spearmanr(diag[metric], diag["mean_val_rank"]).correlation)
        corr_rows.append({"metric": metric, "spearman_vs_val_rank": rho})
    pd.DataFrame(corr_rows).to_csv(SPEARMAN_OUTPUT, index=False)

    summary = {
        "source": str(DIAGNOSTIC_INPUT.relative_to(ROOT)),
        "rows": int(len(diag)),
        "top10_rows": int(diag["is_top10"].sum()),
        "metrics": corr_rows,
        "budgets": BUDGETS.tolist(),
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {SPEARMAN_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {RETENTION_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
