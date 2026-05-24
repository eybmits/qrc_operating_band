#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q scripts
python scripts/run_qrc_phase_grid.py
python scripts/run_qrc_phase_ablation_slices.py
python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
