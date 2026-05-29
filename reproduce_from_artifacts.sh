#!/usr/bin/env bash
set -euo pipefail

python scripts/analyze_phase_map_generalization.py
python scripts/make_figures_and_build_data.py
./paper/build.sh --update-pdf
