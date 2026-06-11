# Reproducibility

This repository provides a complete, versioned reproducibility package for the quantum-reservoir operating-band paper.

Current publication tag: `v1.0.3-publication`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exact reproducibility workflow

From checked-in artifacts, regenerate all manuscript outputs and supporting tables/figures:

```bash
./reproduce_from_artifacts.sh
```

This command:

```bash
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

### Full rerun from scratch

If you need to recompute simulation results from raw runs as well:

```bash
./reproduce.sh
```

## Scope and fixed setup

- Fixed control grid and seeds are defined in `scripts/run_qrc_phase_grid.py` and `scripts/run_qrc_phase_ablation_slices.py`.
- Seeds used in the core experiments: `42` to `61`.
- Core hyperparameters (including ridge search) are selected on validation only; holdout is never used for band selection.
- The repository contains checked-in CSV/JSON result files, the simulator, selection and diagnostic code, figure/table scripts, and the final manuscript assets.

## Included materials

The package includes:

- Checked-in simulation outputs and summaries under `data/`
- Manuscript number macros in `paper/generated/phase_map_numbers.tex`
- Figure outputs in `paper/gfx/`
- Final manuscript in `paper/qrc_phase_diagram.pdf`
- Flat manuscript source upload package in `dist/qrc_operating_band_source.zip`

## Reproduced artifacts

- `paper/qrc_phase_diagram.pdf`
- `paper/generated/phase_map_numbers.tex`
- `paper/gfx/fig1_short_phase_maps.{pdf,png}`
- `paper/gfx/fig4_gamma_slices_compact.{pdf,png}`
- `paper/gfx/fig2_short_evidence.{pdf,png}`
- `paper/gfx/fig3_memory_capacity_screens.{pdf,png}`
- `paper/gfx/gamma_regime_slices_only.{pdf,png}`
