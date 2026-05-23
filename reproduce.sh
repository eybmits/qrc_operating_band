#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q scripts
python scripts/make_figures_and_build_data.py
python scripts/make_qrc_v6_with_real_diagnostics.py
