import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml
from mpc.problems import solution


SEEDS = [1924, 1925, 1926, 1973, 2025,
         2024, 2012, 1958, 1962, 1997]

def encontrar_burgers(obj):
    if isinstance(obj, dict):
        if {"nu", "a", "b", "bounds"}.issubset(obj):
            return obj
        for valor in obj.values():
            achado = encontrar_burgers(valor)
            if achado is not None:
                return achado
    elif isinstance(obj, list):
        for valor in obj:
            achado = encontrar_burgers(valor)
            if achado is not None:
                return achado
    return None


CONFIG = yaml.safe_load(Path(__file__).resolve().parent.joinpath("config.yaml").read_text())
PROBLEM_CFG = encontrar_burgers(CONFIG)

if PROBLEM_CFG is None:
    raise ValueError("Parametros de Burgers nao encontrados em config.yaml")

NU = float(PROBLEM_CFG["nu"])
BOUNDS = torch.tensor(PROBLEM_CFG["bounds"], dtype=torch.float32)

print("Referencia original:", PROBLEM_CFG, flush=True)


def reference():
    """Mesma solucao analitica do experimento supervisionado."""
    def avaliar(tx):
        tx = np.asarray(tx)
        coords = torch.tensor(
            np.column_stack([tx[:, 1], tx[:, 0]]),
            dtype=torch.float32
        )
        with torch.no_grad():
            return solution(
                "burgers", coords, PROBLEM_CFG
            ).numpy().ravel()
    return avaliar


class NormalizeInput(nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.register_buffer("lo", bounds[:, 0])
        self.register_buffer("hi", bounds[:, 1])

    def forward(self, x):
        return 2 * (x - self.lo) / (self.hi - self.lo) - 1


# ============================================================
# MLP clássica
# ============================================================

def make_model(depth, width):
    layers = []
    n = 2

    for _ in range(depth):
        layers.extend([
            nn.Linear(n, width),
            nn.Tanh()
        ])
        n = width

    layers.append(nn.Linear(n, 1))

    return nn.Sequential(*layers)


# ============================================================
# Treinamento PINN
# ============================================================

def train_one(args, depth, seed, ref):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cpu")

    def tensor(a):
        return torch.as_tensor(
            a, dtype=torch.float32, device=device
        )

    # Pontos de colocação
    xf = tensor(rng.uniform(0, 1, (args.points, 1)))
    tf = tensor(rng.uniform(0, 1, (args.points, 1)))

    # Condição inicial
    xi = tensor(rng.uniform(0, 1, (args.ic, 1)))

    ic_input = torch.cat([
        xi,
        torch.zeros(args.ic, 1)
    ], dim=1)

    ic_target = solution("burgers", ic_input, PROBLEM_CFG)

    # Condições de contorno
    tb = tensor(rng.uniform(0, 1, (args.bc, 1)))

    bc_left = torch.cat([
        torch.zeros(args.bc, 1), tb
    ], dim=1)

    bc_right = torch.cat([
        torch.ones(args.bc, 1), tb
    ], dim=1)

    # Validação independente
    valx = rng.uniform(0, 1, (args.val, 1))
    valt = rng.uniform(0, 1, (args.val, 1))

    val_input = tensor(
        np.column_stack([valx.ravel(), valt.ravel()])
    )

    val_target = tensor(
        ref(
            np.column_stack([valt.ravel(), valx.ravel()])
        ).reshape(-1, 1)
    )

    model = nn.Sequential(NormalizeInput(BOUNDS), make_model(depth, args.width))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.epochs // 3),
            gamma=0.5
        )
    else:
        scheduler = None

    run_dir = (
        args.output
        / "results"
        / "burgers"
        / f"pinn_mlp_depth_{depth}"
        / f"seed_{seed}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "epoch",
        "train_mse",
        "val_mse",
        "loss_pde",
        "loss_ic",
        "loss_bc",
        "grad_classical",
        "grad_all",
        "lr"
    ]

    best_val = np.inf
    best_epoch = 0
    best_state = None

    with (run_dir / "history.csv").open(
        "w", newline=""
    ) as stream:

        writer = csv.DictWriter(
            stream, fieldnames=columns
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):

            optimizer.zero_grad(set_to_none=True)

            # Entradas com derivadas automáticas
            pts = torch.cat(
                [xf, tf], dim=1
            ).detach().requires_grad_(True)

            u = model(pts)

            # Primeiras derivadas
            du = torch.autograd.grad(
                u,
                pts,
                torch.ones_like(u),
                create_graph=True
            )[0]

            ux = du[:, :1]
            ut = du[:, 1:2]

            # Segunda derivada espacial
            d2u = torch.autograd.grad(
                ux,
                pts,
                torch.ones_like(ux),
                create_graph=True
            )[0]

            uxx = d2u[:, :1]

            # Resíduo de Burgers
            residual = ut + u * ux - NU * uxx

            loss_pde = residual.square().mean()

            loss_ic = (
                model(ic_input) - ic_target
            ).square().mean()

            loss_bc = (
                model(bc_left).square().mean()
                + model(bc_right).square().mean()
            ) / 2

            # Loss total
            loss = loss_pde + loss_ic + loss_bc

            loss.backward()

            # Norma do gradiente
            grad = float(torch.sqrt(sum(
                p.grad.detach().square().sum()
                for p in model.parameters()
                if p.grad is not None
            )))

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Histórico
            if (
                epoch == 1
                or epoch % args.log_every == 0
                or epoch == args.epochs
            ):

                with torch.no_grad():
                    val_mse = float(
                        (
                            model(val_input) - val_target
                        ).square().mean()
                    )

                if val_mse < best_val:
                    best_val = val_mse
                    best_epoch = epoch
                    best_state = copy.deepcopy(
                        model.state_dict()
                    )

                writer.writerow({
                    "epoch": epoch,
                    "train_mse": float(loss.detach()),
                    "val_mse": val_mse,
                    "loss_pde": float(loss_pde.detach()),
                    "loss_ic": float(loss_ic.detach()),
                    "loss_bc": float(loss_bc.detach()),
                    "grad_classical": grad,
                    "grad_all": grad,
                    "lr": optimizer.param_groups[0]["lr"]
                })

                stream.flush()

            if epoch % 1000 == 0:
                print(
                    f"depth={depth} seed={seed} "
                    f"epoch={epoch} "
                    f"loss={float(loss.detach()):.4e}",
                    flush=True
                )

    # ========================================================
    # Avaliação do melhor modelo
    # ========================================================

    model.load_state_dict(best_state)

    xg = np.linspace(0, 1, args.grid)
    tg = np.linspace(0, 1, args.grid)

    xx, tt = np.meshgrid(xg, tg)

    coords = np.column_stack([
        xx.ravel(), tt.ravel()
    ])

    exact = ref(
        np.column_stack([tt.ravel(), xx.ravel()])
    )

    predictions = []

    with torch.no_grad():
        for chunk in np.array_split(
            coords,
            max(1, int(np.ceil(len(coords) / 4096)))
        ):
            predictions.append(
                model(tensor(chunk)).cpu().numpy().ravel()
            )

    pred = np.concatenate(predictions)
    error = pred - exact

    metrics = {
        "model": f"pinn_mlp_depth_{depth}",
        "seed": seed,
        "depth": depth,
        "width": args.width,
        "epochs": args.epochs,
        "lr": args.lr,
        "points": args.points,
        "ic_points": args.ic,
        "bc_points_each": args.bc,
        "scheduler": args.scheduler,
        "best_epoch": best_epoch,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": float(
            np.linalg.norm(error) / np.linalg.norm(exact)
        ),
        "best_val_mse": best_val
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    torch.save(
        model.state_dict(),
        run_dir / "best.pt"
    )

    print(
        f"FINISHED depth={depth} seed={seed} "
        f"rmse={metrics['rmse']:.6g}",
        flush=True
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "step"],
        default="cosine"
    )

    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--ic", type=int, default=100)
    parser.add_argument("--bc", type=int, default=100)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--grid", type=int, default=250)

    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS
    )

    args = parser.parse_args()

    torch.set_num_threads(2)

    ref = reference()

    for seed in args.seeds:
        train_one(args, args.depth, seed, ref)


if __name__ == "__main__":
    main()
