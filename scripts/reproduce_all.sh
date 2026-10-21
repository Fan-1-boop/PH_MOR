#!/usr/bin/env bash
# Regenerate local Phase 1/2 JSON results and charts from the committed scripts.
# Numerical outputs are not guaranteed identical across BLAS/SciPy versions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$ROOT/phase1_schatten"
python experiment.py
python stress.py
python rankone.py
python intrinsic.py
python margin.py
python tradeoffs.py
python audit.py
cd "$ROOT/phase2_structural_gap"
python reproduce.py
echo "Reproduction finished; see JSON/PNG outputs in each phase directory."
