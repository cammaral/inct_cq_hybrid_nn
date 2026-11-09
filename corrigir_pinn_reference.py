from pathlib import Path

p = Path("pinn_mlp_depth.py")
s = p.read_text()

s = s.replace(
    "from scipy.integrate import solve_ivp\nfrom scipy.interpolate import RegularGridInterpolator",
    "import yaml\nfrom mpc.problems import solution"
)

inicio = s.index("NU = 0.01 / np.pi")
fim = s.index("# ============================================================\n# MLP clássica", inicio)

novo = '''def encontrar_burgers(obj):
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


CONFIG = yaml.safe_load(Path("config.yaml").read_text())
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


'''

s = s[:inicio] + novo + s[fim:]

s = s.replace(
    "ic_target = -torch.sin(np.pi * xi)",
    'ic_target = solution("burgers", ic_input, PROBLEM_CFG)'
)

s = s.replace(
    "model = make_model(depth, args.width)",
    "model = nn.Sequential(NormalizeInput(BOUNDS), make_model(depth, args.width))"
)

p.write_text(s)
print("PINN corrigida para a solucao analitica original.")
