# Cleanup Notes

The source ZIP contained a nested `scripts/all_available_qrc_python/` discovery tree with repeated copies of the same final scripts. The cleaned repository keeps the runnable, final sources directly under `scripts/` and removes redundant nested copies.

Portability changes:

- replaced hard-coded `/mnt/data/...` paths with repository-relative paths;
- kept the QRC simulator as `scripts/qrc_stateful_minimal_suite.py` because diagnostic recomputation imports it directly;
- changed the simulator default output from `/mnt/data/qrc_stateful_run` to `outputs/qrc_stateful_run`;
- kept generated reference figures under `gfx_reference/`;
- added `reproduce.sh` as the default local validation entry point.

The package remains code/data only. Manuscript LaTeX/PDF files were not present in `qrc_final_code_package.zip`.
