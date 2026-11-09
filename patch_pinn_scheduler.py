from pathlib import Path

p = Path("pinn_mlp_depth.py")
s = p.read_text()

old = '''    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )'''

new = '''    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.epochs // 3),
            gamma=0.5
        )
    else:
        scheduler = None'''

if old not in s:
    raise RuntimeError("Bloco do scheduler não encontrado")

s = s.replace(old, new)

old = '''            scheduler.step()'''
new = '''            if scheduler is not None:
                scheduler.step()'''

if old not in s:
    raise RuntimeError("scheduler.step() não encontrado")

s = s.replace(old, new)

old = '''    parser.add_argument("--lr", type=float, default=0.005)'''
new = '''    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "step"],
        default="cosine"
    )'''

if old not in s:
    raise RuntimeError("Argumento --lr não encontrado")

s = s.replace(old, new)

s = s.replace(
    '"scheduler": "cosine",',
    '"scheduler": args.scheduler,'
)

p.write_text(s)

print("Schedulers none, cosine e step habilitados.")
