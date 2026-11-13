#!/usr/bin/env python3
"""PINN-MPC com StronglyEntanglingLayers, 5 qubits, 3 camadas."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import pinn_mpc_serial as base


class StrongZFeatures(nn.Module):
    """Rot + CNOT, equivalente ao StronglyEntanglingLayers.

    Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi).
    CNOT: (q, (q+r) % n), com r = 1, 2, 3.
    Retorna apenas os cinco valores esperados Z.
    """

    def __init__(self, n_qubits, layers, observable):
        super().__init__()

        if (n_qubits, layers, observable.lower()) != (5, 3, "z"):
            raise ValueError(
                "Use 5 qubits, 3 camadas e observaveis Z"
            )

        self.n = n_qubits
        self.layers = layers
        self.observable = "z"

        self.angles = nn.Parameter(
            torch.empty(layers, n_qubits, 3).uniform_(-0.4, 0.4)
        )

        indices = torch.arange(2 ** n_qubits, dtype=torch.long)

        signs = torch.stack([
            (1 - 2 * ((indices >> (n_qubits - q - 1)) & 1)).float()
            for q in range(n_qubits)
        ])

        self.register_buffer("z_signs", signs, persistent=False)

        # Permutacoes CNOT: nenhuma matriz densa de 32x32.
        perms = []

        for layer in range(layers):
            r = (layer % (n_qubits - 1)) + 1

            for control in range(n_qubits):
                target = (control + r) % n_qubits
                control_bit = (
                    indices >> (n_qubits - control - 1)
                ) & 1

                perm = indices ^ (
                    control_bit << (n_qubits - target - 1)
                )
                perms.append(perm)

        self.register_buffer(
            "cnot_perms", torch.stack(perms), persistent=False
        )

    def forward(self):

        state = torch.zeros(
            2 ** self.n,
            dtype=torch.complex64,
            device=self.angles.device
        )
        state[0] = 1

        for layer in range(self.layers):

            for q in range(self.n):
                phi, theta, omega = self.angles[layer, q]

                c = torch.cos(theta / 2)
                s = torch.sin(theta / 2)

                ry = torch.stack((
                    torch.stack((c, -s)),
                    torch.stack((s, c))
                )).to(torch.complex64)

                rz_phi = torch.diag(torch.stack((
                    torch.exp(-0.5j * phi),
                    torch.exp(0.5j * phi)
                )))

                rz_omega = torch.diag(torch.stack((
                    torch.exp(-0.5j * omega),
                    torch.exp(0.5j * omega)
                )))

                rot = rz_omega @ ry @ rz_phi

                state = base.gate_on_qubit(
                    state, rot, q, self.n
                )

            for q in range(self.n):
                state = state[
                    self.cnot_perms[layer * self.n + q]
                ]

        probabilities = state.abs().square()

        return self.z_signs @ probabilities


def self_test():
    """Testar observaveis, gradientes e estrutura."""

    torch.manual_seed(1924)
    model = StrongZFeatures(5, 3, "z")

    values = model()

    assert values.shape == (5,)
    assert torch.isfinite(values).all()
    assert (values.abs() <= 1.00001).all()

    values.square().sum().backward()

    assert torch.isfinite(model.angles.grad).all()
    assert set(model.state_dict()) == {"angles"}

    # Referencia independente usando matrizes completas
    # apenas neste teste pequeno de 5 qubits.
    angles = model.angles.detach().clone().requires_grad_(True)

    state = torch.zeros(32, dtype=torch.complex64)
    state[0] = 1
    indices = torch.arange(32)

    for layer in range(3):
        for q in range(5):
            phi, theta, omega = angles[layer, q]

            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)

            ry = torch.stack((
                torch.stack((c, -s)),
                torch.stack((s, c))
            )).to(torch.complex64)

            rz1 = torch.diag(torch.stack((
                torch.exp(-0.5j * phi),
                torch.exp(0.5j * phi)
            )))
            rz2 = torch.diag(torch.stack((
                torch.exp(-0.5j * omega),
                torch.exp(0.5j * omega)
            )))

            gate = rz2 @ ry @ rz1

            op = torch.ones((1, 1), dtype=torch.complex64)

            for wire in range(5):
                local = (
                    gate if wire == q
                    else torch.eye(2, dtype=torch.complex64)
                )
                op = torch.kron(op, local)

            state = op @ state

        for control in range(5):
            target = (control + layer + 1) % 5
            perm = indices ^ (
                ((indices >> (4 - control)) & 1)
                << (4 - target)
            )
            state = state[perm]

    expected = model.z_signs @ state.abs().square()

    torch.testing.assert_close(
        values, expected, rtol=1e-5, atol=1e-6
    )

    expected.square().sum().backward()

    torch.testing.assert_close(
        model.angles.grad,
        angles.grad,
        rtol=1e-4,
        atol=1e-5
    )

    print("OK: StronglyEntanglingLayers — valores e gradientes")
    print("Qubits: 5 | Camadas: 3 | Parametros quanticos: 45")


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output", type=Path,
        default=Path("runs/pinn/mpc_strongly_entangling")
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int,
        default=[1924, 1925, 1926]
    )

    ap.add_argument("--epochs", type=int, default=15000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--ic", type=int, default=100)
    ap.add_argument("--bc", type=int, default=100)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--grid", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--checkpoint-every", type=int, default=250)

    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    if args.self_test:
        self_test()
        return

    # Substituir somente o circuito desta execucao.
    # Nenhum arquivo do projeto original e modificado.
    base.QuantumFeatures = StrongZFeatures

    stage = "strong_entangling_q5_l3"

    for seed in args.seeds:

        run = (
            args.output / stage
            / "q5_l3_z_polynomial"
            / f"seed_{seed}"
        )

        if args.dry_run:
            print("WOULD RUN:", run)
            continue

        # Reutiliza a loss PINN, Adam, scheduler,
        # checkpoint e avaliacao do codigo original.
        base.train_one(
            args, stage, 5, 3, "z", "polynomial", seed
        )

        metrics_path = run / "metrics.json"

        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())

            metrics["quantum_ansatz"] = (
                "StronglyEntanglingLayers_Rot_CNOT"
            )
            metrics["entangling_ranges"] = [1, 2, 3]
            metrics["quantum_trainable_parameters"] = 45

            temp = metrics_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(metrics, indent=2))
            temp.replace(metrics_path)


if __name__ == "__main__":
    main()
