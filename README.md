# QRC Phase Diagram Code Package

This repository contains the cleaned code and data package for the QRC operating-regime / phase-diagram experiments. It was built from `qrc_final_code_package.zip` and removes the duplicated discovery tree that was present in the archive.

## Contents

- `scripts/make_figures_and_build_data.py`: rebuilds the main paper figure set from the included CSV/JSON tables.
- `scripts/run_qrc96_local_refinement.py`: reruns the frozen 3x3x3 QRC96 Pauli-ring local refinement grid.
- `scripts/run_qrc96_same_arch_expanded.py`: reruns the expanded 5x5x5 same-architecture QRC96 Pauli-ring grid.
- `scripts/run_qrc96_sunspots_fine_refinement.py`: reruns the same-architecture Sunspots fine grid used by the task-wise plateau selector.
- `scripts/analyze_qrc96_esn100.py`: selects QRC96/ESN100 by shared validation and writes paired statistical tests.
- `scripts/analyze_qrc96_esn100_taskwise.py`: selects QRC96/ESN100 task-wise by validation-plateau medoid for the non-floor task comparison.
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
python scripts/run_qrc96_same_arch_expanded.py --overwrite
python scripts/run_qrc96_sunspots_fine_refinement.py --overwrite
python scripts/analyze_qrc96_esn100.py
python scripts/analyze_qrc96_esn100_taskwise.py
```

The expanded same-architecture refinement evaluates 125 QRC96 settings across 10 seeds and 4 tasks. The Sunspots fine refinement evaluates 990 same-architecture QRC96 settings across 10 seeds inside the same operating regime. The shared analysis selects by mean validation NMSE; the task-wise analysis retains configurations within 1% of the best mean validation NMSE and selects the normalized hyperparameter medoid of that validation plateau before reporting holdout statistics. The architecture remains fixed; only `beta`, `lambda`, and `gamma` vary.

To rebuild only the paper PDF:

```bash
./paper/build.sh
```

## Key Saved Results

The saved summary in `data/final_summary_numbers.json` reports:

- QRC96 shared mean holdout NMSE: `0.07694429835846507`
- ESN100 shared mean holdout NMSE: `0.09033359418003997`
- Seed-level paired delta `ESN100 - QRC96`: `0.013389295821574897` with bootstrap 95% CI `[0.008202319188585463, 0.018672818414763283]`
- Task-wise non-floor paired delta on NARMA10/Sunspots: `0.05798159491562421` with bootstrap 95% CI `[0.035752884153071124, 0.08143287738309309]`
- Task-wise Sunspots paired delta: `0.024203395021002223` with bootstrap 95% CI `[0.002575866173436994, 0.05648809020908055]` and `8 / 10` QRC seed wins.
- QRC memory-capacity Spearman correlation: `-0.8528302709370734`
- legacy MC-screen zero-memory points: `922 / 2450`
- current intrinsic-diagnostic zero-memory points used by the paper: `792 / 2450`

See `docs/reproducibility.md` for exact commands and validation notes.
