from pathlib import Path

# ============================================================
# Hiperparâmetros selecionados até agora
# Atualizar após cada ablação
# ============================================================

BEST_EPOCHS = 15000
BEST_LR = 0.005
BEST_SCHED = "cosine"
BEST_POINTS = 1000

ablacoes = {
    "epochs": ["2500", "5000", "10000", "15000", "20000"],
    "lr": ["0.0001", "0.0005", "0.001", "0.005", "0.01"],
    "scheduler": ["none", "cosine", "step"],
    "points": ["250", "500", "1000", "2000", "4000"]
}

for stage, values in ablacoes.items():

    # Distribuição alternada entre os três nós
    nos = [
        values[0::3],
        values[1::3],
        values[2::3]
    ]

    linhas = [
        "#!/bin/bash",
        f"#PBS -N pinn_{stage}",
        "#PBS -q workq",
        "#PBS -l select=3:ncpus=2:mem=4gb",
        "#PBS -l place=scatter",
        "#PBS -l walltime=48:00:00",
        "#PBS -j oe",
        "",
        "set -euo pipefail",
        'cd "$PBS_O_WORKDIR"',
        "",
        (
            'ROOT="$PBS_O_WORKDIR/runs/pinn/'
            f'mlp_{stage}_${{PBS_JOBID%%.*}}"'
        ),
        'mkdir -p "$ROOT"',
        "",
        "export OMP_NUM_THREADS=2",
        "export MKL_NUM_THREADS=2",
        "export OPENBLAS_NUM_THREADS=2",
        "",
        "STATUS=0",
        ""
    ]

    for i, valores_no in enumerate(nos):

        argumentos = " ".join(valores_no)

        linhas.extend([
            f"# Nó {i + 1}",
            (
                f'pbsdsh -n {i} -- /bin/bash '
                f'"$PBS_O_WORKDIR/worker_pinn_ablation.sh" '
                f'"$ROOT" {stage} {BEST_EPOCHS} '
                f'{BEST_LR} {BEST_SCHED} {BEST_POINTS} '
                f'{argumentos} &'
            ),
            f"P{i}=$!",
            ""
        ])

    linhas.extend([
        'wait "$P0" || STATUS=1',
        'wait "$P1" || STATUS=1',
        'wait "$P2" || STATUS=1',
        "",
        'COUNT=$(find "$ROOT" -name metrics.json | wc -l)',
        f'echo "Treinamentos concluídos: $COUNT/{len(values)*10}"',
        "",
        f'if [ "$COUNT" -ne {len(values)*10} ]; then',
        "    STATUS=1",
        "fi",
        "",
        'echo "Status final: $STATUS"',
        'exit "$STATUS"',
        ""
    ])

    arquivo = Path(f"job_pinn_{stage}.pbs")
    arquivo.write_text("\n".join(linhas))

    print(f"Criado: {arquivo}")

