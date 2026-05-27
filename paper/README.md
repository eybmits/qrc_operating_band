# Paper

This directory contains the cleaned QRC-only manuscript package.

- `qrc_phase_diagram.tex`: manuscript source.
- `qrc_phase_diagram.pdf`: reviewed compiled PDF copy.
- `IEEEtran.cls`: local class file kept for reproducible builds.
- `generated/phase_map_numbers.tex`: generated macros for QRC-only phase-map band, transfer, ablation, robustness, and diagnostic numbers.
- `gfx/fig1_short_phase_maps.pdf`: leave-one-task-out phase maps.
- `gfx/gamma_regime_slices_only.pdf`: damping-slice atlas for the validation-ranked phase grid.
- `gfx/fig2_short_evidence.pdf`: validation-band frequency and mechanism-ablation maps.
- `gfx/fig3_memory_capacity_screens.pdf`: memory map, memory-rank relation, and screening-retention figure.

Build the paper from the repository root:

```bash
./paper/build.sh
```

Inside the standalone TeX ZIP, run the local build script instead:

```bash
./build.sh --update-pdf
```

In the repository, the build output goes to `paper/build/`, which is ignored by Git. After checking the built PDF, refresh the tracked copy with:

```bash
./paper/build.sh --update-pdf
```
