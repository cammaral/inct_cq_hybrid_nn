#!/bin/bash
set -euo pipefail

ROOT="$1"
shift

cd "$PBS_O_WORKDIR"

PYTHON="$HOME/miniforge3/envs/mpc/bin/python"

for WIDTH in "$@"; do

    echo "Iniciando width=$WIDTH em $(hostname)"

    "$PYTHON" -u pinn_mlp_depth.py \
        --depth 4 \
        --width "$WIDTH" \
        --epochs 15000 \
        --lr 0.005 \
        --points 1000 \
        --ic 100 \
        --bc 100 \
        --val 300 \
        --grid 250 \
        --output "$ROOT/width_${WIDTH}" \
        > "$ROOT/width_${WIDTH}.log" 2>&1

    echo "Finalizado width=$WIDTH"

done
