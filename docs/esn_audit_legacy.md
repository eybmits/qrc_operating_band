# Archived Optional ESN Audit

Earlier drafts of this project included a QRC96-vs-ESN100 supporting comparison. That material is retained in the repository for provenance, but it is no longer part of the main four-page paper claim.

The current paper is QRC-only. It argues for a transferable memory-defined phase-map operating regime using QRC phase grids, QRC-only mechanism ablations, leave-one-task/seed transfer, and intrinsic diagnostics.

Archived ESN-related artifacts include:

- `scripts/analyze_qrc96_esn100.py`
- `scripts/analyze_qrc96_esn100_taskwise.py`
- `scripts/run_qrc96_same_arch_expanded.py`
- `scripts/run_qrc96_sunspots_fine_refinement.py`
- `data/esn_candidate_performance.csv`
- `data/esn_memory_capacity_recomputed.csv`
- `data/qrc96_esn100_*.csv`
- `data/qrc96_esn100_*.json`
- `paper/generated/qrc96_esn100_numbers.tex`
- `paper/generated/qrc96_taskwise_numbers.tex`

These files can still be inspected or rerun manually, but `./reproduce.sh` intentionally excludes them so that the standard build path matches the QRC-only paper.
