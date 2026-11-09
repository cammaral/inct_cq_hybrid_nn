#!/bin/bash
set -euo pipefail

ROOT="$1"
shift

PYTHON="$HOME/miniforge3/envs/mpc/bin/python"

cd "$PBS_O_WORKDIR"

for NAME in "$@"; do
    echo "Iniciando $NAME em $(hostname)"

    "$PYTHON" -u run.py \
        --config "$ROOT/config_${NAME}.yaml" \
        --problem burgers \
        > "$ROOT/${NAME}.log" 2>&1

    echo "Finalizado $NAME"
done
