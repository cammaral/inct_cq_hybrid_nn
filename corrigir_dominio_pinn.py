from pathlib import Path

p = Path("pinn_mlp_depth.py")
s = p.read_text()

# Pontos de colocação
s = s.replace(
    'rng.uniform(-1, 1, (args.points, 1))',
    'rng.uniform(0, 1, (args.points, 1))'
)

# Condição inicial
s = s.replace(
    'rng.uniform(-1, 1, (args.ic, 1))',
    'rng.uniform(0, 1, (args.ic, 1))'
)

# Contorno esquerdo x=0
s = s.replace(
    '-torch.ones(args.bc, 1), tb',
    'torch.zeros(args.bc, 1), tb'
)

# Validação
s = s.replace(
    'rng.uniform(-1, 1, (args.val, 1))',
    'rng.uniform(0, 1, (args.val, 1))'
)

# Grade de teste
s = s.replace(
    'np.linspace(-1, 1, args.grid)',
    'np.linspace(0, 1, args.grid)'
)

p.write_text(s)

print("Dominio corrigido para [0,1] x [0,1].")
