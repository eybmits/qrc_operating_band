# Data Manifest

## Main Tables

- `qrc_seed_ensemble_grid.csv`: QRC grid evaluations across tasks, seeds, and parameter points.
- `qrc96_local_refinement_grid.csv`: frozen 3x3x3 QRC96 Pauli-ring local refinement grid used for the final comparison.
- `qrc96_local_refinement_metadata.json`: task splits, feature names, grid values, seeds, and ridge grid for the local QRC96 run.
- `qrc96_same_arch_expanded_grid.csv`: expanded 5x5x5 same-architecture QRC96 Pauli-ring grid used for shared ESN100 comparisons and task-wise non-Sunspots selections.
- `qrc96_same_arch_expanded_metadata.json`: architecture lock, task splits, feature names, grid values, seeds, and ridge grid for the expanded QRC96 run.
- `qrc96_sunspots_fine_refinement_grid.csv`: same-architecture Sunspots fine refinement grid used by the task-wise validation-plateau selector.
- `qrc96_sunspots_fine_refinement_metadata.json`: architecture lock, Sunspots split, feature names, grid values, seeds, and ridge grid for the Sunspots fine run.
- `qrc96_esn100_stats.json`: shared-validation selected QRC96/ESN100 configurations, paired bootstrap intervals, Wilcoxon tests, sign tests, and per-task decomposition.
- `qrc96_esn100_taskwise_stats.json`: task-wise validation-plateau selected QRC96/ESN100 configurations and paired statistics.
- `qrc96_esn100_per_task.csv`: per-task selected-config holdout means and QRC96 wins over seeds.
- `qrc96_esn100_taskwise_per_task.csv`: per-task task-wise validation-plateau selected holdout means, selected settings, and QRC96 wins over seeds.
- `qrc96_esn100_seed_pairs.csv`: seed-level task-mean paired holdout deltas.
- `qrc96_esn100_taskwise_seed_pairs.csv`: seed-level paired deltas after task-wise validation-plateau selection.
- `qrc96_esn100_task_seed_pairs.csv`: full task-seed paired holdout table for selected QRC96 and ESN100.
- `qrc96_esn100_taskwise_task_seed_pairs.csv`: full task-seed paired holdout table for task-wise validation-plateau selected QRC96 and ESN100.
- `qrc96_esn100_selected_configs.csv`: validation-selected QRC96 and ESN100 settings.
- `qrc96_esn100_taskwise_selected_configs.csv`: task-wise validation-plateau selected QRC96 and ESN100 settings.
- `qrc96_final10_results.csv`: archived earlier partial QRC96 selected-regime evaluations; retained for provenance, not used by the rebuilt final figures.
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
