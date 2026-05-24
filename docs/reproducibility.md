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
python scripts/run_qrc_phase_grid.py
python scripts/run_qrc_phase_ablation_slices.py
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

Expected outputs include:

- `data/qrc_phase_ablation_slice_grid.csv`
- `data/qrc_phase_ablation_slice_metadata.json`
- `data/qrc_seed_ensemble_grid.csv`
- `data/qrc_architecture_robustness_grid.csv`
- `data/qrc_phase_grid_metadata.json`
- `data/phase_map_generalization_stats.json`
- `data/phase_map_band_membership.csv`
- `data/phase_map_leave_one_task_out.csv`
- `data/phase_map_leave_one_seed_out.csv`
- `data/phase_map_task_seed_transfer_matrix.csv`
- `data/phase_map_holdout_performance.csv`
- `data/phase_map_architecture_robustness.csv`
- `data/phase_map_ablation_retention.csv`
- `paper/generated/phase_map_numbers.tex`
- `paper/gfx/fig1_short_phase_maps.{png,pdf}`
- `paper/gfx/fig2_short_evidence.{png,pdf}`
- `paper/gfx/fig3_memory_capacity_screens.{png,pdf}`
- `paper/build/qrc_phase_diagram.pdf`
- `paper/qrc_phase_diagram.pdf`

The canonical row counts are 19,600 rows for `data/qrc_seed_ensemble_grid.csv`,
19,600 rows for `data/qrc_architecture_robustness_grid.csv`, and 27,440 rows
for `data/qrc_phase_ablation_slice_grid.csv`.

The optional visual-candidate gallery is not part of the standard paper build. Regenerate it manually with:

```bash
python scripts/make_phase_map_candidate_gallery.py
```

## Main Grid and Focused Ablation Recompute

To force recomputation of the canonical 20-seed base grid, the 4q/3-layer robustness grid, and the focused mechanism ablations:

```bash
python scripts/run_qrc_phase_grid.py --overwrite
python scripts/run_qrc_phase_ablation_slices.py --overwrite
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

The main grid evaluates the base 4q/2-layer reservoir and one nearby 4q/3-layer robustness variant across seeds 42--61. The ablation sweep evaluates the QRC16 `Z+ZZ` readout at the `gamma=0.12` slice for the base amplitude-damped RxZZ reservoir and six mechanism controls: `gamma0_amplitude`, `dephasing`, `depolarizing`, `mixer_none`, `mixer_rx_only`, and `mixer_zz_only`.

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

## Implementation Details Now Explicit in the Paper

- Frozen rotations: the main `rx_zz` mixer uses one seed-specific draw
  `theta_i ~ Uniform(0, 2*pi)` from `numpy.random.default_rng(seed)`, reused at
  every time step and layer. Exact seed-42--61 values are checked in at
  `data/frozen_rx_angles.csv`.
- Simulator: the QRC16 experiments use exact complex density matrices for
  `n=4`, column-major vectorization, an exact local channel superoperator, and
  exact expectation values. No finite shots or hardware noise are modeled.
- Readout: the main feature vector concatenates `Z_i` and ring-pair `Z_i Z_j`
  expectations after each of two layers, for 16 features. Ridge readouts include
  an unpenalized intercept.
- Diagnostics: MC/IPC are computed on the common iid uniform diagnostic drive
  with length 900, washout 150, seed 12345, ridge `alpha=1e-8`, and a 70/30
  chronological train/test split. MC uses linear delays 1--10; IPC adds linear
  delays 1--20, second-order Legendre delays 1--10, and the ten cross-delay
  products listed in `scripts/compute_qrc_real_diagnostics.py`.
- Confidence intervals: reported transfer CIs use 50,000 nonparametric
  bootstrap resamples of the mean with seed 20260523 and 2.5/97.5 percentiles.

## Code and Data Availability

All scripts, checked-in CSV/JSON inputs, generated TeX number macros, figure
artifacts, and the built PDF are part of this repository:

<https://github.com/eybmits/qrc_phase_diagram>

## Simulator Smoke Test

For a small standalone run of the simulator, use:

```bash
python scripts/qrc_stateful_minimal_suite.py --grid-mode tiny --task-profile tiny --skip-ablations --outdir outputs/smoke
```

The `outputs/` directory is ignored by Git.

## Archived Optional Baseline

QRC96/ESN100 scripts and data are retained for provenance only. They are documented in `docs/esn_audit_legacy.md` and are not part of the current main-paper reproduction path.
