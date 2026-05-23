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
python scripts/analyze_qrc96_esn100.py
python scripts/make_figures_and_build_data.py
python scripts/make_qrc_v6_with_real_diagnostics.py
./paper/build.sh
```

Expected outputs include:

- `data/qrc96_esn100_stats.json`
- `data/qrc96_esn100_per_task.csv`
- `data/qrc96_esn100_seed_pairs.csv`
- `paper/generated/qrc96_esn100_numbers.tex`
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

## QRC96 Local Refinement Recompute

The final QRC96/ESN100 comparison uses a frozen local QRC96 grid inside the QRC16-discovered operating regime:

```bash
python scripts/run_qrc96_local_refinement.py --overwrite
python scripts/analyze_qrc96_esn100.py
```

This writes:

- `data/qrc96_local_refinement_grid.csv`
- `data/qrc96_local_refinement_metadata.json`
- `data/qrc96_esn100_stats.json`
- `data/qrc96_esn100_per_task.csv`
- `data/qrc96_esn100_seed_pairs.csv`
- `data/qrc96_esn100_task_seed_pairs.csv`
- `data/qrc96_esn100_selected_configs.csv`
- `paper/generated/qrc96_esn100_numbers.tex`

The analysis selects QRC96 and ESN100 only by mean validation NMSE across tasks and seeds, then computes holdout NMSE and paired tests for the selected configurations.

## Simulator Smoke Test

For a small standalone run of the simulator, use:

```bash
python scripts/qrc_stateful_minimal_suite.py --grid-mode tiny --task-profile tiny --skip-ablations --outdir outputs/smoke
```

The `outputs/` directory is ignored by Git.
