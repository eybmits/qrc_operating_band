# Data Manifest

## Main Grids

| File | Purpose |
|---|---|
| `data/qrc_seed_ensemble_grid.csv` | Base 4q/2-layer phase-map grid over four tasks, seeds 42-61, and the `(beta, lambda, gamma)` grid including the low-damping `gamma=0.01` slice. |
| `data/qrc_architecture_robustness_grid.csv` | Nearby 4q/3-layer robustness grid using the same task, seed, readout, split protocol, and damping grid. |
| `data/qrc_phase_ablation_slice_grid.csv` | Focused mechanism ablation sweep at `gamma=0.12`. |
| `data/qrc_phase_grid_metadata.json` | Metadata for the base and robustness grids. |
| `data/qrc_phase_ablation_slice_metadata.json` | Metadata for the ablation grid. |
| `data/frozen_rx_angles.csv` | Frozen seed-specific mixer angles for seeds 42-61. |

## Analysis Outputs

| File | Purpose |
|---|---|
| `data/phase_map_generalization_stats.json` | Primary operating-band, transfer, holdout, ablation, robustness, and diagnostic summaries. |
| `data/phase_map_band_membership.csv` | Membership of each coordinate in the scanned `B_{p,q}` bands. |
| `data/phase_map_leave_one_task_out.csv` | Held-out-task transfer summary. |
| `data/phase_map_leave_one_seed_out.csv` | Held-out-seed transfer summary. |
| `data/phase_map_task_seed_transfer_matrix.csv` | Task-by-seed rank matrix for the primary band. |
| `data/phase_map_holdout_performance.csv` | Holdout audit after validation-only band selection. |
| `data/phase_map_architecture_robustness.csv` | Compact robustness table for the 4q/3-layer variant. |
| `data/phase_map_ablation_retention.csv` | Retention of the base band under mechanism ablations. |
| `data/final_summary_numbers.json` | Small status summary of the final paper numbers. |

## Diagnostics

| File | Purpose |
|---|---|
| `data/qrc_real_current_intrinsic_diagnostics.csv` | Simulator-backed memory and IPC diagnostics. |
| `data/qrc_real_current_intrinsic_diagnostics_with_perf.csv` | Diagnostics merged with validation/test performance summaries. |
| `data/qrc_real_current_diagnostic_spearman.csv` | Raw Spearman diagnostic correlations. |
| `data/qrc_real_current_diagnostic_spearman_named.csv` | Named diagnostic correlations used in the paper. |
| `data/screening_retention_recomputed_intrinsic_diagnostics.csv` | Screening-retention curves for memory, IPC, feature diversity, and random selection. |
| `data/qrc_intrinsic_diagnostic_summary.json` | Diagnostic summary metadata. |

## Paper Artifacts

| File | Purpose |
|---|---|
| `paper/generated/phase_map_numbers.tex` | Generated numeric macros used by the manuscript. |
| `paper/gfx/fig1_short_phase_maps.{pdf,png}` | Transfer phase maps. |
| `paper/gfx/fig4_gamma_slices_compact.{pdf,png}` | Compact 2x2 damping-slice figure included in the manuscript. |
| `paper/gfx/fig2_short_evidence.{pdf,png}` | Operating-band map and mechanism ablation rank-loss maps. |
| `paper/gfx/fig3_memory_capacity_screens.{pdf,png}` | Memory map, memory-rank relation, and screening retention. |
| `paper/gfx/gamma_regime_slices_only.{pdf,png}` | Supplemental wide damping-slice atlas showing the strict core and broader same-consistency band across gamma values. |
| `paper/qrc_phase_diagram.pdf` | Reviewer-facing PDF. |
| `dist/qrc_phase_diagram_tex_package.zip` | Compact TeX submission package. |
