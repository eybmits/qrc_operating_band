#!/usr/bin/env python3
"""Run the canonical 20-seed QRC phase-map grids used by the paper."""
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
GAMMAS = np.array([0.0, 0.05, 0.12, 0.22, 0.30], dtype=float)
SEEDS = list(range(42, 62))
ALPHAS = np.logspace(-8, 3, 12)
READOUT = "Z+ZZ"

SETTINGS: Dict[str, Dict[str, Any]] = {
    "base_4q2layer_z_zz": {
        "description": "4q, 2 layers, Z+ZZ",
        "output": ROOT / "data" / "qrc_seed_ensemble_grid.csv",
        "n": 4,
        "layers": 2,
    },
    "variant_4q3layer_z_zz": {
        "description": "4q, 3 layers, Z+ZZ",
        "output": ROOT / "data" / "qrc_architecture_robustness_grid.csv",
        "n": 4,
        "layers": 3,
    },
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
    return len(SEEDS) * len(BETA_FRACS) * len(LAMBDA_FRACS) * len(GAMMAS) * n_tasks


def existing_complete(path: Path, n_tasks: int = 4) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    keys = ["task", "seed", "readout", "beta_pi", "lambda_pi", "gamma"]
    return len(df) == expected_rows(n_tasks) and df[keys].drop_duplicates().shape[0] == expected_rows(n_tasks)


def write_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_frozen_rx_angles(path: Path) -> None:
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2 * np.pi, size=4)
        for qubit, value in enumerate(theta):
            rows.append({"seed": int(seed), "qubit": int(qubit), "theta_rx_rad": float(value)})
    write_atomic_csv(pd.DataFrame(rows), path)


def evaluate_setting(tasks: Dict[str, Any], setting_key: str, spec: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = len(SEEDS) * len(BETA_FRACS) * len(LAMBDA_FRACS) * len(GAMMAS) * len(tasks)
    done = 0
    t0 = time.time()
    uin_cache: Dict[str, Any] = {}
    for seed in SEEDS:
        for beta_frac in BETA_FRACS:
            for gamma in GAMMAS:
                for lambda_frac in LAMBDA_FRACS:
                    cfg = QRCConfig(
                        n=int(spec["n"]),
                        layers=int(spec["layers"]),
                        beta=float(beta_frac * math.pi),
                        lam=float(lambda_frac * math.pi),
                        gamma=float(gamma),
                        channel="amplitude",
                        topology="ring",
                        mixer="rx_zz",
                        input_mode="uniform",
                        seed=int(seed),
                    )
                    for task in tasks.values():
                        Xz, Xzz = simulate_features(task, cfg, uin_cache)
                        X = make_readout_matrices(Xz, Xzz, seed=int(seed))[READOUT]
                        res = evaluate_readout(X, task.y, task, ALPHAS)
                        rows.append(
                            {
                                "task": task.name,
                                "seed": int(seed),
                                "readout": READOUT,
                                "beta_pi": float(beta_frac),
                                "lambda_pi": float(lambda_frac),
                                "gamma": float(gamma),
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
                        if done % 250 == 0 or done == total:
                            print(
                                f"{setting_key}: {done}/{total} rows, elapsed {time.time() - t0:.1f}s",
                                flush=True,
                            )
    return pd.DataFrame(rows).sort_values(["beta_pi", "lambda_pi", "gamma", "seed", "task"]).reset_index(drop=True)


def run(task_profile: str, overwrite: bool, only: List[str] | None = None) -> None:
    tasks = make_tasks(task_profile)
    selected_settings = {k: v for k, v in SETTINGS.items() if only is None or k in only}
    if not selected_settings:
        raise SystemExit(f"No settings selected from: {', '.join(SETTINGS)}")

    outputs: Dict[str, Dict[str, Any]] = {}
    for setting_key, spec in selected_settings.items():
        output = Path(spec["output"])
        if output.exists() and not overwrite:
            if existing_complete(output, len(tasks)):
                print(f"{output} already contains the complete 20-seed {setting_key} grid; leaving it unchanged.")
                outputs[setting_key] = {
                    "description": spec["description"],
                    "output": str(output.relative_to(ROOT)),
                    "rows": int(expected_rows(len(tasks))),
                }
                continue
            raise SystemExit(f"{output} exists but is incomplete; rerun with --overwrite after inspecting it.")

        df = evaluate_setting(tasks, setting_key, spec)
        if len(df) != expected_rows(len(tasks)):
            raise RuntimeError(f"{setting_key}: expected {expected_rows(len(tasks))} rows, got {len(df)}")
        write_atomic_csv(df, output)
        print(f"wrote {output} ({len(df)} rows)")
        outputs[setting_key] = {
            "description": spec["description"],
            "output": str(output.relative_to(ROOT)),
            "rows": int(len(df)),
            "n": int(spec["n"]),
            "layers": int(spec["layers"]),
        }

    write_frozen_rx_angles(ROOT / "data" / "frozen_rx_angles.csv")
    metadata = {
        "description": "Canonical QRC phase-map grids for the 20-seed paper and one nearby architecture robustness variant.",
        "task_profile": task_profile,
        "tasks": task_metadata(tasks),
        "readout": READOUT,
        "beta_pi_grid": BETA_FRACS.tolist(),
        "lambda_pi_grid": LAMBDA_FRACS.tolist(),
        "gamma_grid": GAMMAS.tolist(),
        "seeds": SEEDS,
        "ridge_alpha_grid": [float(a) for a in ALPHAS],
        "fixed_architecture": {
            "topology": "ring",
            "mixer": "rx_zz",
            "input_mode": "uniform",
            "channel": "amplitude",
            "trained_parameters": "ridge readout only",
        },
        "settings": outputs,
    }
    write_json(metadata, ROOT / "data" / "qrc_phase_grid_metadata.json")
    print(f"wrote {ROOT / 'data' / 'qrc_phase_grid_metadata.json'}")
    print(f"wrote {ROOT / 'data' / 'frozen_rx_angles.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-profile", choices=["tiny", "lite", "paperish"], default="paperish")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", choices=sorted(SETTINGS), nargs="*")
    args = parser.parse_args()
    run(args.task_profile, args.overwrite, args.only or None)


if __name__ == "__main__":
    main()
