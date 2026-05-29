# Reproducibility

This repository reproduces the QRC-only paper figures and tables from checked-in simulation artifacts.
The manuscript PDF is constrained to four QCE workshop pages including references.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One-Command Rebuild

For quick regeneration of manuscript outputs from checked-in artifacts (Figs. 1–4 and Tables I–II), use:

```bash
./reproduce_from_artifacts.sh
```

The script runs:

```bash
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
```

The artifact rebuild path does not rerun the reservoir simulations and instead refreshes the published figures/tables from checked-in CSV/JSON artifacts.

To regenerate the full artifacts from scratch (including simulation runs), use:

```bash
./reproduce.sh
```

## Expected Row Counts

| Artifact | Rows |
|---|---:|
| `data/qrc_seed_ensemble_grid.csv` | 23,520 |
| `data/qrc_architecture_robustness_grid.csv` | 23,520 |
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
- `paper/gfx/fig4_gamma_slices_compact.{pdf,png}`
- `paper/gfx/fig2_short_evidence.{pdf,png}`
- `paper/gfx/fig3_memory_capacity_screens.{pdf,png}`
- `paper/gfx/gamma_regime_slices_only.{pdf,png}` as a supplemental wide repository atlas
