#!/usr/bin/env python3
"""Run the frozen QRC96 local refinement grid used for the final comparison."""
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
    make_tasks,
    simulate_pauli_ring_features,
)


BETA_FRACS = np.array([0.005, 0.010, 0.020], dtype=float)
LAMBDA_FRACS = np.array([0.050, 0.075, 0.100], dtype=float)
GAMMAS = np.array([0.050, 0.080, 0.120], dtype=float)
SEEDS = list(range(42, 52))
ALPHAS = np.logspace(-8, 3, 12)
ARCH = "QRC96_L2_RxZZRx_uniform_pauliRing"


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


def existing_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    expected = len(BETA_FRACS) * len(LAMBDA_FRACS) * len(GAMMAS) * len(SEEDS) * 4
    if len(df) != expected:
        return False
    keys = ["task", "seed", "beta_pi", "lambda_pi", "gamma"]
    return df[keys].drop_duplicates().shape[0] == expected


def run(output: Path, metadata_output: Path, task_profile: str, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        if existing_complete(output):
            print(f"{output} already contains the complete 3x3x3 QRC96 grid; leaving it unchanged.")
            return
        raise SystemExit(f"{output} exists but is incomplete; rerun with --overwrite after inspecting it.")

    tasks = make_tasks(task_profile)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    total = len(SEEDS) * len(BETA_FRACS) * len(tasks) * len(LAMBDA_FRACS) * len(GAMMAS)
    done = 0
    feature_names = None

    for seed in SEEDS:
        for beta_frac in BETA_FRACS:
            for task in tasks.values():
                uin_cache: Dict[str, Any] = {}
                for lambda_frac in LAMBDA_FRACS:
                    for gamma in GAMMAS:
                        cfg = QRCConfig(
                            n=4,
                            layers=2,
                            beta=float(beta_frac * math.pi),
                            lam=float(lambda_frac * math.pi),
                            gamma=float(gamma),
                            channel="amplitude",
                            topology="ring",
                            mixer="rx_zz_rx",
                            input_mode="uniform",
                            seed=int(seed),
                        )
                        X, names = simulate_pauli_ring_features(task, cfg, uin_cache=uin_cache)
                        if feature_names is None:
                            feature_names = list(names)
                        elif list(names) != feature_names:
                            raise RuntimeError("QRC96 feature-name order changed within the run.")
                        if X.shape[1] != 96:
                            raise RuntimeError(f"Expected 96 QRC features, got {X.shape[1]}.")
                        res = evaluate_readout(X, task.y, task, ALPHAS)
                        rows.append(
                            {
                                "arch": ARCH,
                                "task": task.name,
                                "seed": int(seed),
                                "beta_pi": float(beta_frac),
                                "lambda_pi": float(lambda_frac),
                                "gamma": float(gamma),
                                "features": int(X.shape[1]),
                                "val_nmse": float(res["val_nmse"]),
                                "test_nmse": float(res["test_nmse"]),
                                "alpha": float(res["alpha"]),
                                "n": cfg.n,
                                "layers": cfg.layers,
                                "topology": cfg.topology,
                                "mixer": cfg.mixer,
                                "input_mode": cfg.input_mode,
                                "channel": cfg.channel,
                                "task_profile": task_profile,
                            }
                        )
                        done += 1
                        if done % 40 == 0 or done == total:
                            elapsed = time.time() - t0
                            print(f"qrc96 local refinement {done}/{total} task-configs, elapsed {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["beta_pi", "lambda_pi", "gamma", "seed", "task"])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(output)

    metadata = {
        "description": "Frozen 3x3x3 local QRC96 refinement grid inside the QRC16-discovered operating regime.",
        "arch": ARCH,
        "task_profile": task_profile,
        "tasks": task_metadata(tasks),
        "beta_pi_grid": BETA_FRACS.tolist(),
        "lambda_pi_grid": LAMBDA_FRACS.tolist(),
        "gamma_grid": GAMMAS.tolist(),
        "seeds": SEEDS,
        "ridge_alpha_grid": [float(x) for x in ALPHAS],
        "feature_count": 96,
        "feature_names": feature_names,
        "rows": int(len(df)),
        "elapsed_seconds": time.time() - t0,
    }
    metadata_output.write_text(json.dumps(metadata, indent=2))
    print(f"wrote {output} ({len(df)} rows)")
    print(f"wrote {metadata_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "qrc96_local_refinement_grid.csv")
    parser.add_argument(
        "--metadata-output", type=Path, default=ROOT / "data" / "qrc96_local_refinement_metadata.json"
    )
    parser.add_argument("--task-profile", choices=["tiny", "lite", "paperish"], default="paperish")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.output, args.metadata_output, args.task_profile, args.overwrite)


if __name__ == "__main__":
    main()
