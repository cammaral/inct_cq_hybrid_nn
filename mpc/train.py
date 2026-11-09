import csv
"""Treinamento supervisionado comum aos quatro modelos; logs por epoca."""
import copy
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .models import Model
from .problems import dataset


def device_from(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel; use device: cpu ou auto")
    if name not in ("cuda", "cpu"):
        raise ValueError("Este projeto suporta CPU e CUDA")
    return torch.device(name)


def grad_norm(params):
    return math.sqrt(sum(float(p.grad.detach().square().sum())
                         for p in params if p.grad is not None))


def csv_file(path, names, rows):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(rows)


def plot_solution(path, phys, target, estimate, size, problem, model):
    extent = [float(phys[:, 0].min()), float(phys[:, 0].max()),
              float(phys[:, 1].min()), float(phys[:, 1].max())]
    maps = [target.reshape(size, size), estimate.reshape(size, size),
            (estimate-target).abs().reshape(size, size)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
    for ax, arr, name in zip(axes, maps, ["Exata", "Predita", "Erro absoluto"]):
        im = ax.imshow(arr.T, origin="lower", extent=extent, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("x" if problem == "burgers" else "S")
        ax.set_ylabel("t" if problem == "burgers" else "tau")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"{problem} | {model}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(path, rows):
    epochs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(epochs, [max(r[1], 1e-12) for r in rows], label="Treino")
    ax.semilogy(epochs, [max(r[2], 1e-12) for r in rows], label="Validacao")
    ax.set(xlabel="Epoca", ylabel="MSE", title="Evolucao do erro")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_expectations(path, rows, terms, tag):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i in range(terms):
        ax.plot([r[0] for r in rows], [r[1+i] for r in rows], label=f"{tag}{i+1}")
    ax.set(xlabel="Epoca", ylabel="Valor esperado / coeficiente", title="Ramos auxiliares")
    ax.grid(alpha=.2)
    ax.legend(ncol=min(terms, 4))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_one(cfg, problem, variant, seed, folder):
    """Dados identicos para todos os modelos com mesma seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device_from(cfg["training"]["device"])
    train, val, test, physical, grid = dataset(
        problem, cfg["problems"][problem], cfg["data"], seed, device)
    x, target = train
    vx, vy = val
    tx, truth = test
    net = Model(variant).to(device)
    settings = cfg["training"]
    epochs = int(settings["epochs"])
    batch = int(settings.get("batch_size", len(x)))
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=float(settings["lr"]))
    schedule_name = settings.get("scheduler", "none")
    if schedule_name not in ("none", "cosine"):
        raise ValueError("scheduler deve ser 'none' ou 'cosine'")
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=float(settings.get("min_lr", 1e-5)))
        if schedule_name == "cosine" else None)

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    history_path = folder / "history.csv"
    history_file = history_path.open("w", newline="", buffering=1)
    history_writer = csv.writer(history_file)
    history_writer.writerow([
        "epoch", "train_mse", "val_mse",
        "grad_all", "grad_aux", "grad_classical", "grad_quantum", "lr"
    ])
    auxiliary = []
    best_loss, best_epoch, best_weights = float("inf"), None, None
    is_mpc = variant["kind"].startswith("mpc_")
    with torch.no_grad():
        initial = net.coefficients().cpu().tolist() if is_mpc else None

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        indices = torch.randperm(len(x), device=device)
        loss_sum = grad_sum = aux_grad_sum = classical_grad_sum = quantum_grad_sum = 0.0
        steps = 0
        for ix in indices.split(batch):
            optimizer.zero_grad(set_to_none=True)
            loss = (net(x[ix]) - target[ix]).square().mean()  # SEM residuo PDE
            loss.backward()
            grad_sum += grad_norm(net.parameters())
            if hasattr(net, "main"):
                classical_grad_sum += grad_norm(net.main.parameters())
            if is_mpc:
                aux_grad_sum += grad_norm(net.aux.parameters())
                if variant["kind"] == "mpc_hybrid":
                    quantum_grad_sum += grad_norm(net.aux.parameters())
            if settings.get("clip") is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(settings["clip"]))
            optimizer.step()
            loss_sum += loss.item() * len(ix)
            steps += 1
        if scheduler is not None:
            scheduler.step()
        net.eval()
        with torch.no_grad():
            val_mse = (net(vx) - vy).square().mean().item()
            # Uma unica avaliacao independente de x, t por epoca.
            if is_mpc:
                z = net.coefficients().detach().cpu()
                powered = z.pow(net.exponents.cpu()) if net.powers else z
                effective = powered / net.terms if net.aggregation == "mean" else powered
                auxiliary.append([epoch] + z.tolist() + powered.tolist() + effective.tolist())
        history_writer.writerow([
            epoch, loss_sum / len(x), val_mse,
            grad_sum / steps, aux_grad_sum / steps,
            classical_grad_sum / steps, quantum_grad_sum / steps,
            optimizer.param_groups[0]["lr"]
        ])
        if val_mse < best_loss:
            best_loss, best_epoch = val_mse, epoch
            best_weights = copy.deepcopy(net.state_dict())
        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            print(f"  {epoch:5d}/{epochs} | validacao MSE = {val_mse:.3e}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    with torch.no_grad():
        final = net.coefficients().cpu().tolist() if is_mpc else None
    history_file.close()
    torch.save(net.state_dict(), folder / "final.pt")
    net.load_state_dict(best_weights)  # modelo selecionado pelo conjunto de validacao
    net.eval()
    with torch.no_grad():
        prediction = torch.cat([net(part).cpu() for part in tx.split(batch)])
        best_coeff = net.coefficients().cpu().tolist() if is_mpc else None
    actual = truth.cpu()
    error = prediction - actual
    result = {
        "problem": problem, "model": variant["name"], "seed": seed,
        "device": str(device),
        "parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
        "time_s": round(seconds, 3),
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "rmse": float(error.square().mean().sqrt()),
        "mae": float(error.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(error) /
                             torch.linalg.vector_norm(actual).clamp_min(1e-12)),
        "max_abs_error": float(error.abs().max()),
    }
    if problem == "black_scholes":
        result["rmse_price"] = result["rmse"] * cfg["problems"][problem]["strike"]

    with (folder / "metrics.json").open("w") as f:
        json.dump({**result, "model_config": variant,
                   "initial_coefficients": initial,
                   "best_coefficients": best_coeff,
                   "final_coefficients": final}, f, indent=2)
    with history_path.open(newline="") as f:
        history = [
            [float(v) for v in row]
            for row in list(csv.reader(f))[1:]
        ]
    plot_history(folder / "loss.png", history)
    if is_mpc:
        tag = "exp_" if variant["kind"] == "mpc_hybrid" else "z_"
        cols = [f"{tag}{i}" for i in range(1, net.terms + 1)]
        csv_file(folder / ("expectations.csv" if tag == "exp_" else "coefficients.csv"),
                 ["epoch"] + cols + [f"power_{i}" for i in range(1, net.terms + 1)]
                 + [f"effective_{i}" for i in range(1, net.terms + 1)], auxiliary)
        plot_expectations(folder / "auxiliary_evolution.png", auxiliary, net.terms, tag)

    plot_solution(folder / "solution.png", physical, actual.flatten(),
                  prediction.flatten(), grid, problem, variant["name"])
    with (folder / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["coordinate_1", "coordinate_2", "exact", "prediction", "error"])
        writer.writerows(zip(physical[:, 0].tolist(), physical[:, 1].tolist(),
                             actual.flatten().tolist(), prediction.flatten().tolist(),
                             error.flatten().tolist()))
    torch.save(net.state_dict(), folder / "best.pt")
    return result
