# QRC Phase Diagram Publication Package

This repository contains the QRC-only code, data, figures, and paper source for the operating-band phase-diagram manuscript.

Code and data availability: [project repository](https://github.com/eybmits/qrc_phase_diagram)

## Contents

- `reproduce.sh`: canonical one-command rebuild of the reviewer artifacts.
- `requirements.txt`: Python dependencies for the analysis and figure pipeline.
- `scripts/run_qrc_phase_grid.py`: canonical 20-seed base phase-map grid plus the 4q/3-layer robustness grid.
- `scripts/run_qrc_phase_ablation_slices.py`: focused mechanism ablation sweep at the central `gamma=0.12` phase-map slice.
- `scripts/analyze_qrc_intrinsic_diagnostics.py`: QRC-only diagnostic correlations and screening-retention summaries.
- `scripts/analyze_phase_map_generalization.py`: operating-band definitions, leave-one-task/seed transfer, holdout audit, robustness table, bootstrap CIs, and manuscript number macros.
- `scripts/make_figures_and_build_data.py`: paper figures and compact summary numbers from checked-in CSV/JSON artifacts.
- `scripts/compute_qrc_real_diagnostics.py`: slower simulator-backed intrinsic diagnostic recomputation.
- `scripts/compute_qrc_real_diagnostics_chunk.py`: optional chunked diagnostic recomputation; chunk outputs are ignored by Git.
- `scripts/qrc_stateful_minimal_suite.py`: standalone stateful dissipative QRC simulator used by the sweeps and diagnostics.
- `data/`: canonical CSV/JSON tables used by the scripts and manuscript figures.
- `paper/`: LaTeX source, local class file, generated number macros, three paper figures, and the compiled PDF.
- `dist/qrc_phase_diagram_tex_package.zip`: compact TeX submission package with source, figures, build script, and the finished PDF.
- `docs/data_manifest.md` and `docs/reproducibility.md`: reviewer-facing artifact and rebuild notes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Run the standard QRC-only rebuild from the repository root:

```bash
./reproduce.sh
```

This compiles the scripts, verifies or creates the canonical 20-seed phase grids and focused ablation grid, regenerates diagnostic summaries, recomputes phase-map statistics/macros, regenerates all paper figures, and rebuilds the tracked paper PDF.

Expected reviewer artifacts after a clean run:

- `data/qrc_seed_ensemble_grid.csv`: 19,600 base-grid rows.
- `data/qrc_architecture_robustness_grid.csv`: 19,600 nearby-depth robustness rows.
- `data/qrc_phase_ablation_slice_grid.csv`: 27,440 focused ablation rows.
- `data/phase_map_generalization_stats.json`, `data/phase_map_holdout_performance.csv`, and `data/phase_map_architecture_robustness.csv`.
- `data/qrc_real_current_diagnostic_spearman_named.csv` and `data/screening_retention_recomputed_intrinsic_diagnostics.csv`.
- `paper/generated/phase_map_numbers.tex`.
- `paper/gfx/fig1_short_phase_maps.{png,pdf}`, `paper/gfx/fig2_short_evidence.{png,pdf}`, and `paper/gfx/fig3_memory_capacity_screens.{png,pdf}`.
- `paper/qrc_phase_diagram.pdf`.

To force the main phase grids and focused ablation sweep from scratch:

```bash
python scripts/run_qrc_phase_grid.py --overwrite
python scripts/run_qrc_phase_ablation_slices.py --overwrite
python scripts/analyze_qrc_intrinsic_diagnostics.py
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

For the slower full simulator-backed diagnostic recomputation:

```bash
python scripts/compute_qrc_real_diagnostics.py
python scripts/analyze_qrc_intrinsic_diagnostics.py
```

## Key Saved Results

The compact saved summary in `data/final_summary_numbers.json` reports:

- Primary band: `B20_q0p7`.
- Primary band size: `4` connected points.
- Primary band medoid: `(beta/pi, lambda/pi, gamma) = (0.04, 0.10, 0.12)`.
- Leave-one-task held-out validation rank: `0.22219055013309674` with 95% bootstrap CI `[0.2047532148094745, 0.23943242969289166]`.
- Leave-one-seed held-out validation rank: `0.16190889212827989` with 95% bootstrap CI `[0.14625605867346939, 0.17821187439261416]`.
- Final validation-selected band holdout rank: `0.16818877551020409`; see `data/phase_map_holdout_performance.csv`.
- 4q/3-layer robustness band: exists, held-out rank `0.15986151603498544`, overlap with base `0.2222222222222222`; see `data/phase_map_architecture_robustness.csv`.
- QRC memory-capacity Spearman correlation with validation rank: `-0.8771113558573318`.
- QRC IPC-total Spearman correlation with validation rank: `-0.9096600976002228`.

## TeX Submission Package

The source-only submission ZIP is tracked at:

```bash
dist/qrc_phase_diagram_tex_package.zip
```

It contains the paper source, `IEEEtran.cls`, generated macros, the three figure PDFs, a local build script, a small README, and the finished PDF. It intentionally omits raw CSV data and temporary build files.
