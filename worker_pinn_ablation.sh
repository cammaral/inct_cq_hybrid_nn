#!/bin/bash
set -euo pipefail

ROOT="$1"
STAGE="$2"
EPOCHS="$3"
LR="$4"
SCHED="$5"
POINTS="$6"
shift 6

cd "$PBS_O_WORKDIR"

PYTHON="$HOME/miniforge3/envs/mpc/bin/python"

for VALUE in "$@"; do

    E="$EPOCHS"
    L="$LR"
    S="$SCHED"
    P="$POINTS"

    case "$STAGE" in
        epochs)    E="$VALUE" ;;
        lr)        L="$VALUE" ;;
        scheduler) S="$VALUE" ;;
        points)    P="$VALUE" ;;
    esac

    NAME="${STAGE}_${VALUE}"

    echo "Iniciando $NAME em $(hostname)"

    "$PYTHON" -u pinn_mlp_depth.py \
        --depth 4 \
        --width 32 \
        --epochs "$E" \
        --lr "$L" \
        --scheduler "$S" \
        --points "$P" \
        --ic 100 \
        --bc 100 \
        --val 300 \
        --grid 250 \
        --output "$ROOT/$NAME" \
        > "$ROOT/$NAME.log" 2>&1

    echo "Finalizado $NAME"

done
