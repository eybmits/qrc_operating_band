# QRC Phase Diagram

Publication repository for the paper **Where a Quantum Reservoir Works: A Transferable Operating Band**.

## Reproducibility package

Complete, versioned material for this submission is available at:

- Repository: `https://github.com/eybmits/qrc_phase_diagram`
- Artifact tag: `v1.0.0-repro` (this commit)
- DOI: pending (to be added at submission)

This package contains:

- Simulator and experiment code (`scripts/`)
- Fixed configuration and search setup for the published experiments (`scripts/run_qrc_phase_grid.py`, `scripts/run_qrc_phase_ablation_slices.py`)
- Checked-in result files under `data/` (CSV/JSON)
- Figure/table generation scripts (`scripts/analyze_phase_map_generalization.py`, `scripts/analyze_qrc_intrinsic_diagnostics.py`, `scripts/make_figures_and_build_data.py`)
- Manuscript sources and generated outputs under `paper/`
- Submission package under `dist/` (`qrc_phase_diagram_tex_package.zip`)

### Exact reproducibility contract

From checked-in artifacts, regenerate the manuscript figures and LaTeX numbers:

```bash
./reproduce_from_artifacts.sh
```

This runs:

```bash
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

For full re-computation from raw simulations:

```bash
./reproduce.sh
```

For anonymous review, use the same tagged material through the submission review channel.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
./reproduce_from_artifacts.sh
```

## Expected outputs

- `paper/qrc_phase_diagram.pdf`
- `paper/generated/phase_map_numbers.tex`
- `paper/gfx/fig1_short_phase_maps.{pdf,png}`
- `paper/gfx/fig4_gamma_slices_compact.{pdf,png}`
- `paper/gfx/fig2_short_evidence.{pdf,png}`
- `paper/gfx/fig3_memory_capacity_screens.{pdf,png}`
- `paper/gfx/gamma_regime_slices_only.{pdf,png}`

## Repository layout

```text
.
├── data/                      # fixed CSV/JSON outputs
├── docs/                      # reproducibility and manifest documentation
├── scripts/                   # simulation, analysis, and plotting code
├── paper/                     # manuscript sources and compiled outputs
├── dist/                      # submission TeX package
├── reproduce.sh               # full pipeline (includes simulation recompute)
└── reproduce_from_artifacts.sh # fast artifact-based rebuild
```

## Main links

- Paper PDF: [paper/qrc_phase_diagram.pdf](paper/qrc_phase_diagram.pdf)
- Manuscript source: [paper/qrc_phase_diagram.tex](paper/qrc_phase_diagram.tex)
- Data manifest: [docs/data_manifest.md](docs/data_manifest.md)
- Reproducibility guide: [docs/reproducibility.md](docs/reproducibility.md)
- TeX package: `dist/qrc_phase_diagram_tex_package.zip`
