# Reproducibility

This repository reproduces the QRC-only paper figures and tables from checked-in simulation artifacts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One-Command Rebuild

```bash
./reproduce.sh
```

The script runs the canonical pipeline:

```bash
python -m compileall -q scripts
python scripts/run_qrc_phase_grid.py
python scripts/run_qrc_phase_ablation_slices.py
python scripts/analyze_qrc_intrinsic_diagnostics.py
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

## Expected Row Counts

| Artifact | Rows |
|---|---:|
| `data/qrc_seed_ensemble_grid.csv` | 19,600 |
| `data/qrc_architecture_robustness_grid.csv` | 19,600 |
| `data/qrc_phase_ablation_slice_grid.csv` | 27,440 |

## Force Recompute

```bash
python scripts/run_qrc_phase_grid.py --overwrite
python scripts/run_qrc_phase_ablation_slices.py --overwrite
python scripts/analyze_qrc_intrinsic_diagnostics.py
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

For the slower simulator-backed diagnostic recomputation:

```bash
python scripts/compute_qrc_real_diagnostics.py
python scripts/analyze_qrc_intrinsic_diagnostics.py
```

## Simulator Scope

- Reservoir: exact density-matrix simulation, four qubits, ring topology, `rx_zz` mixer, uniform scalar input, amplitude damping.
- Main readout: `Z_i` and ring-pair `Z_i Z_j` expectations over two layers, ridge readout with unpenalized intercept.
- Seeds: `42-61`.
- Main band: validation-defined `B_{20,0.7}`; holdout is evaluated only after the band is fixed.
- Diagnostics: memory capacity and IPC use a common iid diagnostic drive, length 900, washout 150, seed 12345.
- Uncertainty: 50,000 nonparametric bootstrap resamples with fixed seed and percentile confidence intervals.
- No finite shots, calibration drift, or hardware noise are modeled.

## Outputs

The rebuild refreshes:

- `paper/qrc_phase_diagram.pdf`
- `paper/generated/phase_map_numbers.tex`
- `paper/gfx/fig1_short_phase_maps.{pdf,png}`
- `paper/gfx/fig2_short_evidence.{pdf,png}`
- `paper/gfx/fig3_memory_capacity_screens.{pdf,png}`
