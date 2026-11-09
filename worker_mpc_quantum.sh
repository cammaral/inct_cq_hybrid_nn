#!/bin/bash
set -euo pipefail

ROOT="$1"
QUBITS="$2"

PYTHON="$HOME/miniforge3/envs/mpc/bin/python"

cd "$PBS_O_WORKDIR"

for LAYERS in 1 3 5; do
    NAME="mpc_q${QUBITS}_l${LAYERS}"

    echo "Iniciando $NAME em $(hostname)"

    "$PYTHON" -u run.py \
        --config "$ROOT/config_${NAME}.yaml" \
        --problem burgers \
        > "$ROOT/${NAME}.log" 2>&1

    echo "Finalizado $NAME"
done
