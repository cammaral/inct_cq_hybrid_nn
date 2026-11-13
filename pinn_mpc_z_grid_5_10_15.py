#!/usr/bin/env python3
"""Extensao da grade PINN-MPC polinomial com observaveis Z."""

import argparse
from pathlib import Path

import torch
import pinn_mpc_serial as original


# Preservar exatamente a implementacao quantica original.
OriginalQuantumFeatures = original.QuantumFeatures


def z_features_only(qubits, layers, observable):
    """Retorna somente os valores esperados Z solicitados.

    Os buffers de base e observaveis sao deterministicos.
    Eles continuam disponiveis durante o calculo, mas nao
    precisam ser salvos nos checkpoints.
    """
    if observable != "z":
        raise ValueError("Esta grade utiliza apenas observaveis Z.")

    features = OriginalQuantumFeatures(
        qubits, layers, observable
    )

    features._non_persistent_buffers_set.update(
        features._buffers.keys()
    )

    return features


# A PINN original passa a utilizar este construtor
# somente durante a execucao deste novo script.
original.QuantumFeatures = z_features_only


def self_test():
    """Comparar valores esperados e gradientes com o original."""

    for qubits, layers in [(5, 3), (10, 3)]:

        torch.manual_seed(1924)
        antigo = OriginalQuantumFeatures(qubits, layers, "z")

        torch.manual_seed(1924)
        novo = z_features_only(qubits, layers, "z")

        a = antigo()
        b = novo()

        assert torch.allclose(
            a, b, atol=1e-6, rtol=1e-6
        ), "Diferenca nos valores esperados"

        a.sum().backward()
        b.sum().backward()

        assert torch.allclose(
            antigo.angles.grad,
            novo.angles.grad,
            atol=1e-6,
            rtol=1e-6
        ), "Diferenca nos gradientes"

        assert list(novo.state_dict()) == ["angles"]

        print(
            f"OK: {qubits} qubits, {layers} camadas"
        )

    print("Valores esperados e gradientes equivalentes.")
    print("Checkpoints sem buffers desnecessarios.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/pinn/mpc_z_grid_5_10_15")
    )

    parser.add_argument(
        "--qubits",
        type=int,
        choices=[5, 10, 15],
        required=True
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1924, 1925, 1926]
    )

    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--ic", type=int, default=100)
    parser.add_argument("--bc", type=int, default=100)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--grid", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    if args.self_test:
        self_test()
        return

    # 5 qubits, 3 e 5 camadas: ja executados.
    layers_by_qubits = {
        5: [10],
        10: [3, 5, 10],
        15: [3, 5, 10]
    }

    for layers in layers_by_qubits[args.qubits]:
        for seed in args.seeds:

            name = (
                f"q{args.qubits}_l{layers}"
                f"_z_polynomial/seed_{seed}"
            )

            if args.dry_run:
                print("WOULD RUN:", name)
                continue

            print("START:", name, flush=True)

            original.train_one(
                args,
                "z_grid_large",
                args.qubits,
                layers,
                "z",
                "polynomial",
                seed
            )


if __name__ == "__main__":
    main()
