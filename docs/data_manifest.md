# Data Manifest

## Main Tables

- `qrc_seed_ensemble_grid.csv`: QRC grid evaluations across tasks, seeds, and parameter points.
- `qrc96_final10_results.csv`: QRC96 selected-regime evaluations.
- `esn_candidate_performance.csv`: ESN candidate baseline evaluations.
- `final_qrc_esn_comparison.csv`: compact QRC/ESN comparison table used by Fig. 2.
- `minimal_ablations_seeded_summary.csv`: mechanism-control summary used by Fig. 2.
- `qrc_memory_capacity_by_seed_theta.csv`: memory-capacity screen used by Fig. 3.
- `esn_memory_capacity_recomputed.csv`: recomputed ESN memory-capacity screen.

## Intrinsic Diagnostics

- `qrc_real_current_intrinsic_diagnostics.csv`: simulator-backed intrinsic diagnostics for each QRC configuration.
- `qrc_real_current_intrinsic_diagnostics_with_perf.csv`: diagnostics merged with validation/test performance summaries.
- `qrc_real_current_diagnostic_spearman.csv`: Spearman correlations from the full diagnostic recomputation.
- `qrc_real_current_diagnostic_spearman_named.csv`: named diagnostic correlations used in the final Fig. 3 text/plot.
- `screening_retention_recomputed_intrinsic_diagnostics.csv`: retained-top-decile curves for memory and intrinsic-capacity metrics.
- `diag_parts/*.csv`: chunk outputs that reproduce the diagnostic table.

The older `qrc_memory_capacity_by_seed_theta.csv` memory-capacity screen has `922 / 2450` zero-MC points. The current intrinsic diagnostic table used by the paper has `792 / 2450` zero-MC points; both values are recorded in `data/final_summary_numbers.json` under separate keys.

## Reference Figures

The `paper/gfx/` directory contains PNG and PDF versions of the rebuilt figure outputs. They are tracked because they are small enough for normal Git use and serve as visual reference artifacts for the saved data.
