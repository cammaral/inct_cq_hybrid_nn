#!/bin/bash
set -euo pipefail

ROOT="$1"
shift

cd "$PBS_O_WORKDIR"

PYTHON="$HOME/miniforge3/envs/mpc/bin/python"

for DEPTH in "$@"; do

    echo "Iniciando depth=$DEPTH em $(hostname)"

    "$PYTHON" -u pinn_mlp_depth.py \
        --depth "$DEPTH" \
        --output "$ROOT" \
        > "$ROOT/depth_${DEPTH}.log" 2>&1

    echo "Finalizado depth=$DEPTH"

done
