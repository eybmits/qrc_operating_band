# Reproducibility Notes

## Environment

Tested with Python 3 and the dependencies in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Standard QRC-Only Rebuild

The current paper is rebuilt from QRC-only phase-map artifacts:

```bash
./reproduce.sh
```

This runs:

```bash
python -m compileall -q scripts
python scripts/run_qrc_phase_ablation_slices.py
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh
```

Expected outputs include:

- `data/qrc_phase_ablation_slice_grid.csv`
- `data/qrc_phase_ablation_slice_metadata.json`
- `data/phase_map_generalization_stats.json`
- `data/phase_map_band_membership.csv`
- `data/phase_map_leave_one_task_out.csv`
- `data/phase_map_leave_one_seed_out.csv`
- `data/phase_map_task_seed_transfer_matrix.csv`
- `data/phase_map_ablation_retention.csv`
- `paper/generated/phase_map_numbers.tex`
- `paper/gfx/fig1_short_phase_maps.{png,pdf}`
- `paper/gfx/fig2_short_evidence.{png,pdf}`
- `paper/gfx/fig3_memory_capacity_screens.{png,pdf}`
- `paper/build/qrc_phase_diagram.pdf`

Use this command to refresh the tracked PDF copy after reviewing a clean build:

```bash
./paper/build.sh --update-pdf
```

## Focused Ablation Recompute

To force recomputation of the focused mechanism ablations:

```bash
python scripts/run_qrc_phase_ablation_slices.py --overwrite
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

The ablation sweep evaluates the QRC16 `Z+ZZ` readout at the `gamma=0.12` slice for the base amplitude-damped RxZZ reservoir and six mechanism controls: `gamma0_amplitude`, `dephasing`, `depolarizing`, `mixer_none`, `mixer_rx_only`, and `mixer_zz_only`.

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

## Archived Optional Baseline

QRC96/ESN100 scripts and data are retained for provenance only. They are documented in `docs/esn_audit_legacy.md` and are not part of the current main-paper reproduction path.
