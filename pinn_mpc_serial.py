#!/usr/bin/env python3
"""Quatro etapas sequenciais PINN-MPC para Burgers, com retomada por seed."""

import argparse
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEEDS = [1924, 1925, 1926, 1973, 2025, 2024, 2012, 1958, 1962, 1997]

FIELDS = [
    "epoch", "train_mse", "val_mse",
    "loss_pde", "loss_ic", "loss_bc",
    "grad_classical", "grad_quantum", "grad_all", "lr"
]


# ============================================================
# 1. Circuito quântico diferenciável
# ============================================================

def gate_on_qubit(state, gate, q, n):
    axes = [q] + [i for i in range(n) if i != q]
    inverse = [axes.index(i) for i in range(n)]
    shaped = state.reshape([2] * n).permute(axes).reshape(2, -1)
    return (gate @ shaped).reshape([2] * n).permute(inverse).reshape(-1)


class QuantumFeatures(nn.Module):

    def __init__(self, n_qubits, layers, observable):
        super().__init__()

        self.n = n_qubits
        self.layers = layers
        self.observable = observable.lower()

        self.angles = nn.Parameter(
            torch.empty(layers, n_qubits, 2).uniform_(-0.4, 0.4)
        )

        self.register_buffer(
            "basis",
            torch.arange(2 ** n_qubits, dtype=torch.long)
        )

        labels = []

        for i in range(n_qubits):
            j = (i + 1) % n_qubits

            if self.observable == "z":
                labels.append([(i, "Z")])

            elif self.observable in ("zz", "xx", "yy"):
                p = self.observable.upper()[0]
                labels.append([(i, p), (j, p)])

            elif self.observable == "mixed":
                p = ("X", "Y", "Z")[i % 3]
                labels.append([(i, p), (j, p)])

            else:
                raise ValueError(
                    f"Observável desconhecido: {observable}"
                )

        for i, label in enumerate(labels):

            flip = torch.zeros(
                2 ** n_qubits, dtype=torch.long
            )

            phase = torch.ones(
                2 ** n_qubits, dtype=torch.complex64
            )

            for q, op in label:

                bit = (
                    self.basis >> (n_qubits - q - 1)
                ) & 1

                if op in ("X", "Y"):
                    flip = flip ^ (1 << (n_qubits - q - 1))

                if op == "Y":
                    phase = phase * (
                        1j * (2 * bit - 1).to(torch.float32)
                    )

                if op == "Z":
                    phase = phase * (
                        1 - 2 * bit
                    ).to(torch.float32)

            self.register_buffer(f"flip_{i}", self.basis ^ flip)
            self.register_buffer(f"phase_{i}", phase)

    def forward(self):

        n = self.n

        state = torch.zeros(
            2 ** n,
            device=self.angles.device,
            dtype=torch.complex64
        )

        state[0] = 1

        for layer in range(self.layers):

            for q in range(n):

                theta, phi = self.angles[layer, q]

                ct = torch.cos(theta / 2)
                st = torch.sin(theta / 2)

                ry = torch.stack((
                    torch.stack((ct, -st)),
                    torch.stack((st, ct))
                )).to(torch.complex64)

                state = gate_on_qubit(state, ry, q, n)

                rz = torch.diag(
                    torch.stack((
                        torch.exp(-0.5j * phi),
                        torch.exp(0.5j * phi)
                    ))
                )

                state = gate_on_qubit(state, rz, q, n)

            # Emaranhamento CZ em cadeia
            for q in range(n - 1):

                mask = (
                    ((self.basis >> (n - q - 1)) & 1)
                    &
                    ((self.basis >> (n - q - 2)) & 1)
                ).to(state.real.dtype)

                state = state * (1 - 2 * mask)

        values = []

        for i in range(n):

            flip = getattr(self, f"flip_{i}")
            phase = getattr(self, f"phase_{i}")

            expectation = (
                state.conj() * phase * state[flip]
            ).sum().real

            values.append(expectation)

        return torch.stack(values)


# ============================================================
# 2. PINN híbrida MPC
# ============================================================

class MPCPINN(nn.Module):

    def __init__(
        self, n_qubits, qlayers, observable,
        coupling, depth=4, width=32
    ):
        super().__init__()

        layers = []
        n_in = 2

        for _ in range(depth):
            layers.extend([
                nn.Linear(n_in, width),
                nn.Tanh()
            ])
            n_in = width

        layers.append(nn.Linear(n_in, n_qubits))

        self.classical = nn.Sequential(*layers)

        self.quantum = QuantumFeatures(
            n_qubits, qlayers, observable
        )

        self.coupling = coupling

    def coefficients(self):

        features = self.quantum()

        if self.coupling == "polynomial":

            indices = torch.arange(
                1, len(features) + 1,
                device=features.device
            )

            return features.pow(indices)

        if self.coupling == "product":
            return features.cumprod(dim=0)

        raise ValueError(self.coupling)

    def forward(self, coords, coefficients=None):

        # Coordenadas físicas [0,1] -> [-1,1]
        y = self.classical(2 * coords - 1)

        if coefficients is None:
            coefficients = self.coefficients()

        return (
            (1 + y) * coefficients
        ).mean(dim=1, keepdim=True)


# ============================================================
# 3. Solução analítica de Burgers
# ============================================================

def exact(coords, nu=0.05, a=0.9, b=1.0):

    x = coords[:, :1]
    t = coords[:, 1:2]

    decay = torch.exp(
        -nu * math.pi ** 2 * t
    )

    return (
        2 * nu * a * math.pi
        * decay * torch.sin(math.pi * x)
        / (b + a * decay * torch.cos(math.pi * x))
    )


# ============================================================
# 4. Amostragem
# ============================================================

def samples(seed, nf, nic, nbc, nv, grid):

    gen = torch.Generator(
        device="cpu"
    ).manual_seed(seed + 2026)

    def random_points(n):
        return torch.rand(n, 2, generator=gen)

    interior = random_points(nf)

    ic = torch.cat((
        torch.rand(nic, 1, generator=gen),
        torch.zeros(nic, 1)
    ), dim=1)

    bc0 = torch.cat((
        torch.zeros(nbc, 1),
        torch.rand(nbc, 1, generator=gen)
    ), dim=1)

    bc1 = torch.cat((
        torch.ones(nbc, 1),
        torch.rand(nbc, 1, generator=gen)
    ), dim=1)

    valid = random_points(nv)

    axes = (
        torch.linspace(0, 1, grid),
        torch.linspace(0, 1, grid)
    )

    xx, tt = torch.meshgrid(
        *axes, indexing="ij"
    )

    test = torch.stack((
        xx.flatten(), tt.flatten()
    ), dim=1)

    return interior, ic, bc0, bc1, valid, test


# ============================================================
# 5. Loss física
# ============================================================

def loss_parts(model, interior, ic, bc0, bc1):

    coords = interior.detach().requires_grad_(True)

    coefficients = model.coefficients()

    u = model(coords, coefficients)

    grad = torch.autograd.grad(
        u, coords,
        torch.ones_like(u),
        create_graph=True
    )[0]

    ux = grad[:, :1]
    ut = grad[:, 1:2]

    grad2 = torch.autograd.grad(
        ux, coords,
        torch.ones_like(ux),
        create_graph=True
    )[0]

    uxx = grad2[:, :1]

    residual = ut + u * ux - 0.05 * uxx

    loss_pde = residual.square().mean()

    loss_ic = (
        model(ic, coefficients) - exact(ic)
    ).square().mean()

    loss_bc = (
        model(bc0, coefficients).square().mean()
        +
        model(bc1, coefficients).square().mean()
    ) / 2

    return loss_pde, loss_ic, loss_bc


# ============================================================
# 6. Avaliação
# ============================================================

def evaluate(model, coords, target, batch=4096):

    model.eval()
    pieces = []

    with torch.no_grad():
        for part in coords.split(batch):
            pieces.append(model(part))

    prediction = torch.cat(pieces, dim=0)
    diff = prediction - target

    rmse = diff.square().mean().sqrt().item()
    mae = diff.abs().mean().item()
    relative_l2 = (
        diff.norm() / target.norm()
    ).item()

    model.train()

    return rmse, mae, relative_l2


def save_atomic(obj, path):

    temp = path.with_name(path.name + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


# ============================================================
# 7. Treinar uma configuração e uma semente
# ============================================================

def train_one(
    args, stage, nq, qlayers,
    observable, coupling, seed
):

    run = (
        args.output / stage
        / f"q{nq}_l{qlayers}_{observable}_{coupling}"
        / f"seed_{seed}"
    )

    metrics_file = run / "metrics.json"

    if metrics_file.exists():
        print(f"SKIP concluído: {run}", flush=True)
        return

    run.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MPCPINN(
        nq, qlayers, observable, coupling,
        depth=args.depth,
        width=args.width
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    interior, ic, bc0, bc1, valid, test = samples(
        seed,
        args.points,
        args.ic,
        args.bc,
        args.val,
        args.grid
    )

    yvalid = exact(valid)
    ytest = exact(test)

    checkpoint = run / "checkpoint.pt"
    history = run / "history.csv"
    best_file = run / "best.pt"

    start = 1
    best_mse = float("inf")
    best_epoch = 0

    # Retomar treinamento interrompido
    if checkpoint.exists():

        ck = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False
        )

        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])

        start = ck["epoch"] + 1
        best_mse = ck["best_mse"]
        best_epoch = ck["best_epoch"]

        if history.exists():

            with history.open(newline="") as f:
                previous = [
                    r for r in csv.DictReader(f)
                    if int(r["epoch"]) <= ck["epoch"]
                ]

            with history.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=FIELDS
                )
                writer.writeheader()
                writer.writerows(previous)

        print(
            f"RESUME {run} from epoch {start}",
            flush=True
        )

    if not history.exists():
        with history.open("w", newline="") as f:
            csv.DictWriter(
                f, fieldnames=FIELDS
            ).writeheader()

    # ========================================================
    # Loop de treinamento
    # ========================================================

    for epoch in range(start, args.epochs + 1):

        optimizer.zero_grad(set_to_none=True)

        pde, initial, boundary = loss_parts(
            model, interior, ic, bc0, bc1
        )

        total = pde + initial + boundary
        total.backward()

        registrar = (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        )

        if registrar:

            gc = sum(
                float(p.grad.square().sum())
                for p in model.classical.parameters()
                if p.grad is not None
            ) ** 0.5

            gq = sum(
                float(p.grad.square().sum())
                for p in model.quantum.parameters()
                if p.grad is not None
            ) ** 0.5

            ga = (gc ** 2 + gq ** 2) ** 0.5

            lr_current = optimizer.param_groups[0]["lr"]

        optimizer.step()
        scheduler.step()

        if registrar:

            with torch.no_grad():

                model.eval()

                val_mse = (
                    model(valid) - yvalid
                ).square().mean().item()

                model.train()

            if val_mse < best_mse:

                best_mse = val_mse
                best_epoch = epoch

                save_atomic(
                    model.state_dict(),
                    best_file
                )

            with history.open("a", newline="") as f:

                writer = csv.DictWriter(
                    f, fieldnames=FIELDS
                )

                writer.writerow({
                    "epoch": epoch,
                    "train_mse": total.item(),
                    "val_mse": val_mse,
                    "loss_pde": pde.item(),
                    "loss_ic": initial.item(),
                    "loss_bc": boundary.item(),
                    "grad_classical": gc,
                    "grad_quantum": gq,
                    "grad_all": ga,
                    "lr": lr_current
                })

            if (
                epoch % (args.log_every * 20) == 0
                or epoch == args.epochs
            ):

                print(
                    f"{stage} q={nq} layers={qlayers} "
                    f"obs={observable} coupling={coupling} "
                    f"seed={seed} epoch={epoch} "
                    f"loss={total.item():.4e} "
                    f"val={val_mse:.4e}",
                    flush=True
                )

        if (
            epoch % args.checkpoint_every == 0
            or epoch == args.epochs
        ):

            save_atomic({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mse": best_mse,
                "best_epoch": best_epoch
            }, checkpoint)

    # ========================================================
    # Avaliação final
    # ========================================================

    model.load_state_dict(
        torch.load(
            best_file,
            map_location="cpu",
            weights_only=True
        )
    )

    rmse, mae, rel = evaluate(
        model, test, ytest
    )

    metrics = {
        "stage": stage,
        "model": "pinn_mpc",
        "seed": seed,
        "depth": args.depth,
        "width": args.width,
        "n_qubits": nq,
        "layers": qlayers,
        "observable": observable,
        "coupling": coupling,
        "epochs": args.epochs,
        "lr": args.lr,
        "scheduler": "cosine",
        "points": args.points,
        "ic_points": args.ic,
        "bc_points_each": args.bc,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "best_epoch": best_epoch,
        "best_val_mse": best_mse,
        "rmse": rmse,
        "mae": mae,
        "relative_l2": rel
    }

    tmp = metrics_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    os.replace(tmp, metrics_file)

    checkpoint.unlink(missing_ok=True)

    print(
        f"DONE {run} rmse={rmse:.6e}",
        flush=True
    )


# ============================================================
# 8. Planejamento das quatro etapas
# ============================================================

def plans():

    # Etapa 1: qubits, com 3 camadas
    s1 = [
        (q, 3, "z", "polynomial")
        for q in (3, 5, 7)
    ]

    # Etapa 2: completar a grade 3 x 3
    s2 = [
        (q, layers, "z", "polynomial")
        for q in (3, 5, 7)
        for layers in (1, 5)
    ]

    # Etapa 3: observáveis, 5 qubits e 3 camadas
    s3 = [
        (5, 3, obs, "polynomial")
        for obs in ("z", "zz", "xx", "yy", "mixed")
    ]

    # Etapa 4: tipos de MPC, 5 qubits e 3 camadas
    s4 = [
        (5, 3, "mixed", coupling)
        for coupling in ("polynomial", "product")
    ]

    return (
        ("01_qubits", s1),
        ("02_layers", s2),
        ("03_observables", s3),
        ("04_couplings", s4)
    )


# ============================================================
# 9. Execução serial
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--output", type=Path, required=True)
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

    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS
    )

    ap.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "01_qubits",
            "02_layers",
            "03_observables",
            "04_couplings"
        ]
    )

    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    args.output.mkdir(
        parents=True, exist_ok=True
    )

    for stage, experiments in plans():

        if args.stage != "all" and stage != args.stage:
            continue

        print(
            f"=== {stage}: "
            f"{len(experiments) * len(args.seeds)} "
            f"execuções ===",
            flush=True
        )

        for nq, layers, obs, coupling in experiments:

            for seed in args.seeds:

                source_stage = None

                if (
                    stage == "03_observables"
                    and obs == "z"
                ):
                    source_stage = "01_qubits"

                elif (
                    stage == "04_couplings"
                    and coupling == "polynomial"
                ):
                    source_stage = "03_observables"

                # Reaproveitar configurações idênticas
                if source_stage is not None and args.stage == "all":

                    name = (
                        f"q{nq}_l{layers}_{obs}_{coupling}"
                    )

                    source = (
                        args.output / source_stage
                        / name / f"seed_{seed}"
                    )

                    target = (
                        args.output / stage
                        / name / f"seed_{seed}"
                    )

                    if (
                        not (target / "metrics.json").exists()
                        and (source / "metrics.json").exists()
                    ):

                        shutil.copytree(
                            source, target,
                            dirs_exist_ok=True
                        )

                        mfile = target / "metrics.json"
                        data = json.loads(
                            mfile.read_text()
                        )

                        data["stage"] = stage

                        mfile.write_text(
                            json.dumps(data, indent=2)
                        )

                        print(
                            f"REUSE {source_stage} -> {stage}: "
                            f"{name}, seed={seed}",
                            flush=True
                        )

                    if (target / "metrics.json").exists():
                        continue

                train_one(
                    args, stage, nq, layers,
                    obs, coupling, seed
                )

    print(
        "TODAS AS ETAPAS SOLICITADAS CONCLUÍDAS",
        flush=True
    )


if __name__ == "__main__":
    main()
