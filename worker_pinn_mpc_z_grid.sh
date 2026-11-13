#!/bin/bash
set -euo pipefail

QUBITS="$1"

cd "$PBS_O_WORKDIR"

PYTHON="$HOME/miniforge3/envs/mpc/bin/python"
OUT="$PBS_O_WORKDIR/runs/pinn/mpc_z_grid_5_10_15"
JOB="${PBS_JOBID%%.*}"

mkdir -p "$OUT"

LOG="$OUT/q${QUBITS}_${JOB}.log"
STATUS_FILE="$OUT/q${QUBITS}_${JOB}.status"

# Evitar considerar um status antigo.
rm -f "$STATUS_FILE"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

{
    echo "Inicio: $(date)"
    echo "Host: $(hostname)"
    echo "Qubits: $QUBITS"
    echo "PBS: $PBS_JOBID"
    echo "================================"
} > "$LOG"

set +e

timeout --signal=TERM --kill-after=60s 46h \
    "$PYTHON" -u pinn_mpc_z_grid_5_10_15.py \
    --output "$OUT" \
    --qubits "$QUBITS" \
    --seeds 1924 1925 1926 \
    --epochs 15000 \
    --depth 4 \
    --width 32 \
    --lr 0.005 \
    --points 1000 \
    --ic 100 \
    --bc 100 \
    --val 300 \
    --grid 250 \
    --log-every 25 \
    --checkpoint-every 250 \
    >> "$LOG" 2>&1

STATUS=$?

set -e

echo "Fim: $(date) | Status: $STATUS" >> "$LOG"

# Registrar o resultado real, mesmo se pbsdsh
# nao propagar corretamente o codigo de saida.
printf '%s\n' "$STATUS" > "${STATUS_FILE}.tmp"
mv "${STATUS_FILE}.tmp" "$STATUS_FILE"

exit "$STATUS"
