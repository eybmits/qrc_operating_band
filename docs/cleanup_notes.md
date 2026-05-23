# Cleanup Notes

The source ZIP contained a nested `scripts/all_available_qrc_python/` discovery tree with repeated copies of the same final scripts. The cleaned repository keeps the runnable, final sources directly under `scripts/` and removes redundant nested copies.

Portability changes:

- replaced hard-coded `/mnt/data/...` paths with repository-relative paths;
- kept the QRC simulator as `scripts/qrc_stateful_minimal_suite.py` because diagnostic recomputation imports it directly;
- changed the simulator default output from `/mnt/data/qrc_stateful_run` to `outputs/qrc_stateful_run`;
- renamed the paper source from `main.tex` to `paper/qrc_phase_diagram.tex`;
- moved generated reference figures to `paper/gfx/`;
- avoided duplicating the paper ZIP's `data/` directory because the root `data/` directory is canonical;
- added `reproduce.sh` as the default local validation entry point.

The initial code package was code/data only. The companion paper package is now integrated under `paper/`.
