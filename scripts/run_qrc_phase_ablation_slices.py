#!/usr/bin/env python3
"""Run focused QRC-only phase-map ablations at the central gamma slice."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qrc_stateful_minimal_suite import (  # noqa: E402
    QRCConfig,
    evaluate_readout,
    make_readout_matrices,
    make_tasks,
    simulate_features,
)


BETA_FRACS = np.array([0.0, 0.02, 0.04, 0.07, 0.10, 0.22, 0.50], dtype=float)
LAMBDA_FRACS = np.array([0.0, 0.05, 0.10, 0.12, 0.16, 0.28, 0.42], dtype=float)
NOMINAL_GAMMA = 0.12
SEEDS = list(range(42, 52))
ALPHAS = np.logspace(-8, 3, 12)
READOUT = "Z+ZZ"

VARIANTS: Dict[str, Dict[str, Any]] = {
    "base_amplitude_rxzz": {"channel": "amplitude", "mixer": "rx_zz", "effective_gamma": NOMINAL_GAMMA},
    "gamma0_amplitude": {"channel": "amplitude", "mixer": "rx_zz", "effective_gamma": 0.0},
    "dephasing": {"channel": "dephasing", "mixer": "rx_zz", "effective_gamma": NOMINAL_GAMMA},
    "depolarizing": {"channel": "depolarizing", "mixer": "rx_zz", "effective_gamma": NOMINAL_GAMMA},
    "mixer_none": {"channel": "amplitude", "mixer": "none", "effective_gamma": NOMINAL_GAMMA},
    "mixer_rx_only": {"channel": "amplitude", "mixer": "rx_only", "effective_gamma": NOMINAL_GAMMA},
    "mixer_zz_only": {"channel": "amplitude", "mixer": "zz_only", "effective_gamma": NOMINAL_GAMMA},
}


def task_metadata(tasks: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    return {
        name: {
            "washout": int(task.washout),
            "train": int(task.train),
            "val": int(task.val),
            "test": int(task.test),
            "total": int(task.total),
        }
        for name, task in tasks.items()
    }


def expected_rows(n_tasks: int = 4) -> int:
    return len(VARIANTS) * len(BETA_FRACS) * len(LAMBDA_FRACS) * len(SEEDS) * n_tasks


def existing_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    keys = ["variant", "task", "seed", "beta_pi", "lambda_pi", "gamma"]
    return len(df) == expected_rows() and df[keys].drop_duplicates().shape[0] == expected_rows()


def base_rows_from_canonical() -> pd.DataFrame:
    path = ROOT / "data" / "qrc_seed_ensemble_grid.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    d = df[
        (df["readout"] == READOUT)
        & np.isclose(df["gamma"], NOMINAL_GAMMA)
        & df["beta_pi"].isin(BETA_FRACS)
        & df["lambda_pi"].isin(LAMBDA_FRACS)
    ].copy()
    if len(d) != len(BETA_FRACS) * len(LAMBDA_FRACS) * len(SEEDS) * 4:
        return pd.DataFrame()
    d.insert(0, "variant", "base_amplitude_rxzz")
    d["effective_gamma"] = NOMINAL_GAMMA
    d["nominal_gamma"] = NOMINAL_GAMMA
    d["readout"] = READOUT
    cols = [
        "variant",
        "task",
        "seed",
        "readout",
        "beta_pi",
        "lambda_pi",
        "gamma",
        "effective_gamma",
        "nominal_gamma",
        "n",
        "layers",
        "topology",
        "mixer",
        "input_mode",
        "channel",
        "val_nmse",
        "test_nmse",
        "alpha",
    ]
    return d[cols]


def evaluate_variant_rows(tasks: Dict[str, Any], variant: str, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(SEEDS) * len(BETA_FRACS) * len(tasks) * len(LAMBDA_FRACS)
    done = 0
    t0 = time.time()
    for seed in SEEDS:
        for beta_frac in BETA_FRACS:
            for task in tasks.values():
                uin_cache: Dict[str, Any] = {}
                for lambda_frac in LAMBDA_FRACS:
                    cfg = QRCConfig(
                        n=4,
                        layers=2,
                        beta=float(beta_frac * math.pi),
                        lam=float(lambda_frac * math.pi),
                        gamma=float(spec["effective_gamma"]),
                        channel=str(spec["channel"]),
                        topology="ring",
                        mixer=str(spec["mixer"]),
                        input_mode="uniform",
                        seed=int(seed),
                    )
                    Xz, Xzz = simulate_features(task, cfg, uin_cache)
                    X = make_readout_matrices(Xz, Xzz, seed=int(seed))[READOUT]
                    res = evaluate_readout(X, task.y, task, ALPHAS)
                    rows.append(
                        {
                            "variant": variant,
                            "task": task.name,
                            "seed": int(seed),
                            "readout": READOUT,
                            "beta_pi": float(beta_frac),
                            "lambda_pi": float(lambda_frac),
                            "gamma": NOMINAL_GAMMA,
                            "effective_gamma": float(cfg.gamma),
                            "nominal_gamma": NOMINAL_GAMMA,
                            "n": int(cfg.n),
                            "layers": int(cfg.layers),
                            "topology": cfg.topology,
                            "mixer": cfg.mixer,
                            "input_mode": cfg.input_mode,
                            "channel": cfg.channel,
                            "val_nmse": float(res["val_nmse"]),
                            "test_nmse": float(res["test_nmse"]),
                            "alpha": float(res["alpha"]),
                        }
                    )
                    done += 1
                    if done % 100 == 0 or done == total:
                        print(
                            f"{variant}: {done}/{total} configs, elapsed {time.time() - t0:.1f}s",
                            flush=True,
                        )
    return rows


def run(output: Path, metadata_output: Path, task_profile: str, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        if existing_complete(output):
            print(f"{output} already contains the complete phase-ablation slice grid; leaving it unchanged.")
            return
        raise SystemExit(f"{output} exists but is incomplete; rerun with --overwrite after inspecting it.")

    tasks = make_tasks(task_profile)
    frames: List[pd.DataFrame] = []
    base = base_rows_from_canonical()
    if base.empty:
        frames.append(pd.DataFrame(evaluate_variant_rows(tasks, "base_amplitude_rxzz", VARIANTS["base_amplitude_rxzz"])))
        base_source = "simulated"
    else:
        frames.append(base)
        base_source = "reused from data/qrc_seed_ensemble_grid.csv at gamma=0.12"

    for variant, spec in VARIANTS.items():
        if variant == "base_amplitude_rxzz":
            continue
        frames.append(pd.DataFrame(evaluate_variant_rows(tasks, variant, spec)))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["variant", "beta_pi", "lambda_pi", "seed", "task"]).reset_index(drop=True)
    if len(df) != expected_rows(len(tasks)):
        raise RuntimeError(f"Expected {expected_rows(len(tasks))} rows, got {len(df)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(output)

    metadata = {
        "description": "Focused QRC-only phase-map ablations at the central gamma=0.12 slice.",
        "base_source": base_source,
        "task_profile": task_profile,
        "tasks": task_metadata(tasks),
        "readout": READOUT,
        "beta_pi_grid": BETA_FRACS.tolist(),
        "lambda_pi_grid": LAMBDA_FRACS.tolist(),
        "nominal_gamma": NOMINAL_GAMMA,
        "seeds": SEEDS,
        "ridge_alpha_grid": [float(a) for a in ALPHAS],
        "variants": VARIANTS,
        "architecture": {
            "n": 4,
            "layers": 2,
            "topology": "ring",
            "input_mode": "uniform",
            "trained_parameters": "ridge readout only",
        },
        "rows": int(len(df)),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output} ({len(df)} rows)")
    print(f"wrote {metadata_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "qrc_phase_ablation_slice_grid.csv")
    parser.add_argument(
        "--metadata-output", type=Path, default=ROOT / "data" / "qrc_phase_ablation_slice_metadata.json"
    )
    parser.add_argument("--task-profile", choices=["tiny", "lite", "paperish"], default="paperish")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.output, args.metadata_output, args.task_profile, args.overwrite)


if __name__ == "__main__":
    main()
