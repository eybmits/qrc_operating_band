# Data Manifest

## Canonical QRC Tables

- `data/qrc_seed_ensemble_grid.csv`: main QRC16 phase-map grid across four tasks, twenty seeds, and the coarse `(beta, lambda, gamma)` grid; 19,600 rows.
- `data/qrc_phase_grid_metadata.json`: task splits, grid values, seeds, ridge grid, base architecture, and nearby-depth robustness setting for the canonical phase-grid run.
- `data/qrc_architecture_robustness_grid.csv`: nearby architecture variant using four qubits, three layers, the same `Z+ZZ` readout, and the same 20-seed protocol; 19,600 rows.
- `data/qrc_phase_ablation_slice_grid.csv`: focused QRC-only mechanism ablation sweep at the central `gamma=0.12` slice; 27,440 rows.
- `data/qrc_phase_ablation_slice_metadata.json`: task splits, grid values, seeds, ridge grid, and variant definitions for the ablation sweep.
- `data/frozen_rx_angles.csv`: exact per-seed frozen `Rx` mixer angles for seeds 42-61 in the main QRC16 simulations.

## Generated Analysis Tables

- `data/phase_map_generalization_stats.json`: primary band summary, connectedness, leave-one-task/seed transfer, bootstrap CIs, ablation retention, holdout audit, robustness table, and diagnostic correlations.
- `data/phase_map_band_membership.csv`: membership of all grid points in `B_{p,q}` bands for `p={10,15,20,25,30}` and `q={0.5,0.6,0.7,0.8}`.
- `data/phase_map_leave_one_task_out.csv`: leave-one-task band-transfer summary for the primary `B_{20,0.7}` protocol.
- `data/phase_map_leave_one_seed_out.csv`: leave-one-seed band-transfer summary for the primary `B_{20,0.7}` protocol.
- `data/phase_map_task_seed_transfer_matrix.csv`: task-by-seed validation ranks of the primary band.
- `data/phase_map_holdout_performance.csv`: final post-selection holdout NMSE and rank audit for the validation-defined primary band and its medoid.
- `data/phase_map_architecture_robustness.csv`: compact robustness table comparing the base 4q/2-layer band with the 4q/3-layer band.
- `data/phase_map_ablation_retention.csv`: retention of the primary base band under focused QRC-only ablations.
- `data/final_summary_numbers.json`: compact summary consumed by quick status checks.

## Intrinsic Diagnostics

- `data/qrc_real_current_intrinsic_diagnostics.csv`: simulator-backed intrinsic diagnostics for each QRC configuration.
- `data/qrc_real_current_intrinsic_diagnostics_with_perf.csv`: diagnostics merged with validation/test performance summaries.
- `data/qrc_real_current_diagnostic_spearman.csv`: Spearman correlations from the full diagnostic recomputation.
- `data/qrc_real_current_diagnostic_spearman_named.csv`: named diagnostic correlations used in the paper.
- `data/screening_retention_recomputed_intrinsic_diagnostics.csv`: retained-top-decile curves for memory and intrinsic-capacity metrics.
- `data/qrc_intrinsic_diagnostic_summary.json`: small metadata file written by the QRC-only diagnostic summary script.

## Manuscript Artifacts

- `paper/generated/phase_map_numbers.tex`: generated numeric macros used by `paper/qrc_phase_diagram.tex`.
- `paper/gfx/fig1_short_phase_maps.{png,pdf}`: leave-one-task-out phase maps.
- `paper/gfx/fig2_short_evidence.{png,pdf}`: validation-band frequency map plus mechanism ablation rank-loss maps.
- `paper/gfx/fig3_memory_capacity_screens.{png,pdf}`: diagnostic correlation and screening-retention figure.
- `paper/qrc_phase_diagram.pdf`: reviewer-facing PDF rebuilt by `./reproduce.sh`.
- `dist/qrc_phase_diagram_tex_package.zip`: compact TeX submission package.
