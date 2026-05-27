# QRC Phase Diagram

Publication repository for the QRC-only short paper:

**Where a Quantum Reservoir Works: A Transferable Operating Band**

Code and data availability: [project repository](https://github.com/eybmits/qrc_phase_diagram)

## Paper

- [PDF](paper/qrc_phase_diagram.pdf)
- TeX submission ZIP: `dist/qrc_phase_diagram_tex_package.zip`

## Figures

![Fig. 1](paper/gfx/fig1_short_phase_maps.png)

![Fig. 2](paper/gfx/fig4_gamma_slices_compact.png)

![Fig. 3](paper/gfx/fig2_short_evidence.png)

![Fig. 4](paper/gfx/fig3_memory_capacity_screens.png)

Extra repository figure:

![Damping-slice atlas](paper/gfx/gamma_regime_slices_only.png)

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh
```

`./reproduce.sh` rebuilds the checked-in CSV/JSON summaries, the four paper figures, the supplemental wide damping atlas, generated TeX numbers, and the 4-page `paper/qrc_phase_diagram.pdf`.

## Main Artifacts

- `data/qrc_seed_ensemble_grid.csv`: base 4q/2-layer QRC phase grid, 19,600 rows.
- `data/qrc_architecture_robustness_grid.csv`: 4q/3-layer robustness grid, 19,600 rows.
- `data/qrc_phase_ablation_slice_grid.csv`: mechanism ablation slice grid, 27,440 rows.
- `data/phase_map_generalization_stats.json`: operating-band, transfer, holdout, robustness, and ablation statistics.
- `data/qrc_real_current_intrinsic_diagnostics.csv`: intrinsic memory/IPC diagnostics.
- `paper/generated/phase_map_numbers.tex`: generated manuscript numbers.
- `paper/gfx/`: final figure PDFs and PNG previews.

## Docs

- [Reproducibility](docs/reproducibility.md)
- [Data manifest](docs/data_manifest.md)
