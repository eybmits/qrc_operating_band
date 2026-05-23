# QRC Phase Diagram Code Package

This repository contains the cleaned code and data package for the QRC operating-regime / phase-diagram experiments. It was built from `qrc_final_code_package.zip` and removes the duplicated discovery tree that was present in the archive.

## Contents

- `scripts/make_figures_and_build_data.py`: rebuilds the main paper figure set from the included CSV/JSON tables.
- `scripts/run_qrc96_local_refinement.py`: reruns the frozen 3x3x3 QRC96 Pauli-ring local refinement grid.
- `scripts/analyze_qrc96_esn100.py`: selects QRC96/ESN100 by shared validation and writes paired statistical tests.
- `scripts/make_qrc_v6_with_real_diagnostics.py`: recomputes the intrinsic-diagnostic ranking tables and the Fig. 3 diagnostic panels from the included diagnostic table.
- `scripts/compute_qrc_real_diagnostics.py`: full intrinsic-diagnostic recomputation from the QRC simulator and saved grid.
- `scripts/compute_qrc_real_diagnostics_chunk.py`: chunked version of the intrinsic-diagnostic recomputation.
- `scripts/qrc_stateful_minimal_suite.py`: standalone stateful dissipative QRC simulator used by the diagnostic recomputation.
- `data/`: curated CSV/JSON tables used by the scripts and manuscript figures.
- `data/diag_parts/`: chunk outputs from the intrinsic-diagnostic recomputation.
- `paper/qrc_phase_diagram.tex`: renamed LaTeX source for the manuscript.
- `paper/qrc_phase_diagram.pdf`: compiled manuscript PDF.
- `paper/build.sh`: reproducible local LaTeX build script.
- `paper/gfx/`: tracked manuscript figures rebuilt from the included tables.
- `docs/`: reproducibility notes, data manifest, and cleanup notes.

The paper source was integrated from `qrc_final_paper_package (1).zip`. Duplicate paper-package copies of `data/` were not retained; `data/` at the repository root is the canonical data directory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Run the lightweight repo check and rebuild the derived tables/figures:

```bash
./reproduce.sh
```

For the full simulator-backed diagnostic recomputation, run:

```bash
python scripts/compute_qrc_real_diagnostics.py
```

That full run evaluates 2,450 QRC configurations and is much slower than the figure-only rebuild. For split execution, use `scripts/compute_qrc_real_diagnostics_chunk.py START END` and write chunks into `data/diag_parts/`.

To rerun the frozen QRC96 local-refinement grid used for the final readout-dimension ESN100 comparison:

```bash
python scripts/run_qrc96_local_refinement.py --overwrite
python scripts/analyze_qrc96_esn100.py
```

The local refinement evaluates 27 QRC96 settings across 10 seeds and 4 tasks, then selects one setting by mean validation NMSE before reporting holdout statistics.

To rebuild only the paper PDF:

```bash
./paper/build.sh
```

## Key Saved Results

The saved summary in `data/final_summary_numbers.json` reports:

- QRC96 shared mean holdout NMSE: `0.07768925554710517`
- ESN100 shared mean holdout NMSE: `0.09033359418003997`
- Seed-level paired delta `ESN100 - QRC96`: `0.01264433863293481` with bootstrap 95% CI `[0.007527776569199887, 0.017961484896988997]`
- QRC memory-capacity Spearman correlation: `-0.8528302709370734`
- legacy MC-screen zero-memory points: `922 / 2450`
- current intrinsic-diagnostic zero-memory points used by the paper: `792 / 2450`

See `docs/reproducibility.md` for exact commands and validation notes.
