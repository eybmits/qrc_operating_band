# QRC Phase Diagram Code Package

This repository contains the cleaned code and data package for the QRC operating-regime / phase-diagram experiments. It was built from `qrc_final_code_package.zip` and removes the duplicated discovery tree that was present in the archive.

## Contents

- `scripts/make_figures_and_build_data.py`: rebuilds the main figure set from the included CSV/JSON tables.
- `scripts/make_qrc_v6_with_real_diagnostics.py`: recomputes the intrinsic-diagnostic ranking tables and the Fig. 3 diagnostic panels from the included diagnostic table.
- `scripts/compute_qrc_real_diagnostics.py`: full intrinsic-diagnostic recomputation from the QRC simulator and saved grid.
- `scripts/compute_qrc_real_diagnostics_chunk.py`: chunked version of the intrinsic-diagnostic recomputation.
- `scripts/qrc_stateful_minimal_suite.py`: standalone stateful dissipative QRC simulator used by the diagnostic recomputation.
- `data/`: curated CSV/JSON tables used by the scripts and manuscript figures.
- `data/diag_parts/`: chunk outputs from the intrinsic-diagnostic recomputation.
- `gfx_reference/`: reference PNG/PDF figures rebuilt from the included tables.
- `docs/`: reproducibility notes, data manifest, and cleanup notes.

The manuscript LaTeX/PDF package is not part of this repository; the original archive notes that it is distributed separately as `qrc_final_paper_package.zip`.

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

## Key Saved Results

The saved summary in `data/final_summary_numbers.json` reports:

- QRC96 shared mean holdout NMSE: `0.07268684590654748`
- ESN100 shared mean holdout NMSE: `0.09033359418003997`
- QRC memory-capacity Spearman correlation: `-0.8528302709370734`
- zero-memory diagnostic points: `922 / 2450`

See `docs/reproducibility.md` for exact commands and validation notes.
