#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

pdflatex -interaction=nonstopmode -halt-on-error -output-directory build qrc_phase_diagram.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build qrc_phase_diagram.tex

if [[ "${1:-}" == "--update-pdf" ]]; then
  cp build/qrc_phase_diagram.pdf qrc_phase_diagram.pdf
fi
