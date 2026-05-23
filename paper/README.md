# Paper

This directory contains the cleaned manuscript package integrated from `qrc_final_paper_package (1).zip`.

- `qrc_phase_diagram.tex`: renamed manuscript source.
- `qrc_phase_diagram.pdf`: reviewed compiled PDF copy.
- `IEEEtran.cls`: local class file kept for reproducible builds.
- `references.bib`: bibliography source from the paper package.
- `gfx/`: manuscript figures generated from the repository root `data/` tables.

Build the paper from this directory or from the repository root:

```bash
./paper/build.sh
```

The build output goes to `paper/build/`, which is ignored by Git. After checking the built PDF, refresh the tracked copy with:

```bash
./paper/build.sh --update-pdf
```
