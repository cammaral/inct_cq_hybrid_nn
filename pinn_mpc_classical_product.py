#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pinn_mpc_serial import (
    SEEDS, exact, samples, loss_parts, evaluate, save_atomic
)

STOP = False

def stop_handler(signum, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, stop_handler)

FIELDS = [
    "epoch", "train_mse", "val_mse",
    "loss_pde", "loss_ic", "loss_bc",
    "grad_classical", "grad_generator",
    "grad_all", "lr"
]


# ============================================================
# Modelo MPC com dois blocos clássicos
# ============================================================

def make_mlp(n_in, n_out, depth, width):
    layers = []

    for _ in range(depth):
        layers += [nn.Linear(n_in, width), nn.Tanh()]
        n_in = width

    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


class ClassicalProductMPC(nn.Module):

    def __init__(self, gen_layers, gen_width):
        super().__init__()

        # Mesma MLP principal da PINN-MPC quântica
        self.classical = make_mlp(2, 5, 4, 32)

        # Substitui somente o circuito quântico
        self.generator = make_mlp(
            1, 5, gen_layers, gen_width
        )

        self.register_buffer(
            "constant", torch.ones(1, 1)
        )

    def coefficients(self):
        c = torch.tanh(
            self.generator(self.constant)
        ).flatten()

        # Acoplamento por produto cumulativo
        return c.cumprod(dim=0)

    def forward(self, coords, coefficients=None):

        y = self.classical(2 * coords - 1)

        if coefficients is None:
            coefficients = self.coefficients()

        return (
            (1 + y) * coefficients
        ).mean(dim=1, keepdim=True)


def grad_norm(parameters):
    return sum(
        p.grad.detach().square().sum().item()
        for p in parameters
        if p.grad is not None
    ) ** 0.5


# ============================================================
# Treinamento
# ============================================================

def train(args, layers, width, seed):

    name = f"layers_{layers}_width_{width}"
    run = args.output / name / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)

    if (run / "metrics.json").exists():
        print(f"SKIP {name} seed={seed}", flush=True)
        return True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ClassicalProductMPC(layers, width)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    interior, ic, bc0, bc1, valid, test = samples(
        seed, args.points, args.ic,
        args.bc, args.val, args.grid
    )

    yvalid = exact(valid)
    ytest = exact(test)

    history = run / "history.csv"
    checkpoint = run / "checkpoint.pt"
    best_file = run / "best.pt"

    start = 1
    best_val = float("inf")
    best_epoch = 0

    # Retomar se já houver checkpoint
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
        best_val = ck["best_val"]
        best_epoch = ck["best_epoch"]

        if history.exists():
            with history.open(newline="") as f:
                rows = [
                    row for row in csv.DictReader(f)
                    if int(row["epoch"]) <= ck["epoch"]
                ]

            with history.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=FIELDS
                )
                writer.writeheader()
                writer.writerows(rows)

        print(
            f"RESUME {name} seed={seed} epoch={start}",
            flush=True
        )

    if not history.exists():
        with history.open("w", newline="") as f:
            csv.DictWriter(
                f, fieldnames=FIELDS
            ).writeheader()

    for epoch in range(start, args.epochs + 1):

        optimizer.zero_grad(set_to_none=True)

        pde, initial, boundary = loss_parts(
            model, interior, ic, bc0, bc1
        )

        total = pde + initial + boundary
        total.backward()

        record = (
            epoch == 1
            or epoch % 25 == 0
            or epoch == args.epochs
        )

        if record:
            gc = grad_norm(model.classical.parameters())
            gg = grad_norm(model.generator.parameters())
            ga = (gc**2 + gg**2)**0.5
            lr = optimizer.param_groups[0]["lr"]

        optimizer.step()
        scheduler.step()

        if record:

            model.eval()
            with torch.no_grad():
                val_mse = (
                    model(valid) - yvalid
                ).square().mean().item()
            model.train()

            if val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch
                save_atomic(model.state_dict(), best_file)

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
                    "grad_generator": gg,
                    "grad_all": ga,
                    "lr": lr
                })

            if epoch % 500 == 0 or epoch == args.epochs:
                print(
                    f"{name} seed={seed} "
                    f"epoch={epoch} "
                    f"loss={total.item():.3e} "
                    f"val={val_mse:.3e}",
                    flush=True
                )

        if (
            epoch % 250 == 0
            or epoch == args.epochs
            or STOP
        ):
            save_atomic({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch
            }, checkpoint)

        if STOP:
            print(f"STOP {name} epoch={epoch}", flush=True)
            return False

    # Avaliação na mesma grade da MPC quântica
    model.load_state_dict(
        torch.load(
            best_file,
            map_location="cpu",
            weights_only=True
        )
    )

    rmse, mae, relative_l2 = evaluate(
        model, test, ytest
    )

    metrics = {
        "model": "pinn_mpc_classical_product",
        "seed": seed,
        "gen_layers": layers,
        "gen_width": width,
        "n_coefficients": 5,
        "coupling": "product",
        "depth": 4,
        "width": 32,
        "epochs": args.epochs,
        "lr": args.lr,
        "scheduler": "cosine",
        "points": args.points,
        "parameters_generator": sum(
            p.numel() for p in model.generator.parameters()
        ),
        "parameters_classical": sum(
            p.numel() for p in model.classical.parameters()
        ),
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "rmse": rmse,
        "mae": mae,
        "relative_l2": relative_l2
    }

    tmp = run / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=2))
    os.replace(tmp, run / "metrics.json")

    checkpoint.unlink(missing_ok=True)

    print(
        f"DONE {name} seed={seed} "
        f"params_generator={metrics['parameters_generator']} "
        f"rmse={rmse:.6e}",
        flush=True
    )

    return True


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--ic", type=int, default=100)
    parser.add_argument("--bc", type=int, default=100)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--grid", type=int, default=250)

    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS
    )

    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    args.output.mkdir(parents=True, exist_ok=True)

    for layers in [1, 2, 3]:
        for width in [1, 2, 3]:
            for seed in args.seeds:

                completed = train(args, layers, width, seed)

                if not completed:
                    return 75

    print("GRADE 3x3 CONCLUÍDA", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
