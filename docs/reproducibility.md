# Reproducibility Notes

## Environment

Tested with Python 3 and the dependencies in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lightweight Rebuild

The standard check compiles all scripts and rebuilds the included derived tables and reference figures:

```bash
./reproduce.sh
```

This runs:

```bash
python -m compileall -q scripts
python scripts/make_figures_and_build_data.py
python scripts/make_qrc_v6_with_real_diagnostics.py
./paper/build.sh
```

Expected outputs include:

- `data/final_summary_numbers.json`
- `data/screening_retention_recomputed_intrinsic_diagnostics.csv`
- `data/qrc_real_current_diagnostic_spearman_named.csv`
- `paper/gfx/fig1_operating_regime_fixed_gamma.{png,pdf}`
- `paper/gfx/fig2_regime_beats_esn_controls.{png,pdf}`
- `paper/gfx/fig3_memory_capacity_screens.{png,pdf}`
- `paper/gfx/fig3d_screening_retention_real_intrinsic.{png,pdf}`
- `paper/build/qrc_phase_diagram.pdf`

Use this command to refresh the tracked PDF copy after reviewing a clean build:

```bash
./paper/build.sh --update-pdf
```

## Full Diagnostic Recompute

The simulator-backed diagnostic recomputation evaluates all unique configurations in `data/qrc_seed_ensemble_grid.csv`:

```bash
python scripts/compute_qrc_real_diagnostics.py
```

It writes:

- `data/qrc_real_current_intrinsic_diagnostics.csv`
- `data/qrc_real_current_intrinsic_diagnostics_with_perf.csv`
- `data/qrc_real_current_diagnostic_spearman.csv`

The same computation can be split into chunks:

```bash
python scripts/compute_qrc_real_diagnostics_chunk.py 0 300
python scripts/compute_qrc_real_diagnostics_chunk.py 300 600
```

Chunk outputs are written under `data/diag_parts/`.

## Simulator Smoke Test

For a small standalone run of the simulator, use:

```bash
python scripts/qrc_stateful_minimal_suite.py --grid-mode tiny --task-profile tiny --skip-ablations --outdir outputs/smoke
```

The `outputs/` directory is ignored by Git.
