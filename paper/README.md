# Paper

This directory contains the cleaned QRC-only manuscript package.

- `qrc_phase_diagram.tex`: four-page manuscript source.
- `qrc_phase_diagram.pdf`: reviewed compiled PDF copy.
- `IEEEtran.cls`: local class file kept for reproducible builds.
- `references.bib`: bibliography source from the original paper package.
- `generated/phase_map_numbers.tex`: generated macros for QRC-only phase-map band, transfer, ablation, and diagnostic numbers.
- `gfx/`: manuscript figures generated from the repository root `data/` tables.

Build the paper from the repository root:

```bash
./paper/build.sh
```

The build output goes to `paper/build/`, which is ignored by Git. After checking the built PDF, refresh the tracked copy with:

```bash
./paper/build.sh --update-pdf
```

Archived QRC96/ESN100 generated macros remain in `generated/` for provenance but are not included by the current paper source.
