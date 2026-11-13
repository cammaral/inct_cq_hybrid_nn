from pathlib import Path
import ast
import shutil

path = Path("mpc/models.py")
source = path.read_text()

# Backup da versão atual
backup = Path("mpc/models_before_fast_z_patch.py")
if not backup.exists():
    shutil.copy2(path, backup)

# Criar ZCircuit, caso ainda não exista
if "class ZCircuit(Circuit):" not in source:
    anchor = "\n\nclass Model(nn.Module):"
    assert source.count(anchor) == 1

    new_class = '''

class ZCircuit(Circuit):
    """Circuito otimizado para observaveis Z locais."""

    def __init__(self, qubits, layers, observables, trainable=True):
        nn.Module.__init__(self)

        self.qubits = qubits
        self.layers = layers
        self.theta = nn.Parameter(
            0.4 * torch.randn(layers, qubits, 2),
            requires_grad=trainable
        )

        for word in observables:
            assert len(word) == qubits
            assert word.count("Z") == 1
            assert set(word) <= {"I", "Z"}

        indices = torch.arange(2 ** qubits)

        diagonals = []
        for word in observables:
            wire = word.index("Z")
            bit = (indices >> (qubits - wire - 1)) & 1
            diagonals.append((1 - 2 * bit).float())

        self.register_buffer(
            "z_diagonals", torch.stack(diagonals)
        )

        signs = []
        for wire in range(qubits - 1):
            left = (indices >> (qubits - wire - 1)) & 1
            right = (indices >> (qubits - wire - 2)) & 1
            signs.append(
                (1 - 2 * left * right).to(torch.complex64)
            )

        self.register_buffer(
            "cz_signs",
            torch.stack(signs) if signs else
            torch.empty(0, 2 ** qubits, dtype=torch.complex64)
        )

    def forward(self, x=None):
        assert x is None

        state = torch.zeros(
            1, 2 ** self.qubits,
            dtype=torch.complex64,
            device=self.theta.device
        )
        state[:, 0] = 1

        for layer in range(self.layers):
            for wire in range(self.qubits):
                angle = self.theta[layer, wire, 0].expand(1)
                phase = self.theta[layer, wire, 1].expand(1)

                c = torch.cos(angle / 2)
                s = torch.sin(angle / 2)

                ry = torch.stack((
                    torch.stack((c, -s), -1),
                    torch.stack((s, c), -1)
                ), -2).to(torch.complex64)

                rz = torch.diag_embed(
                    torch.stack((
                        torch.exp(-0.5j * phase),
                        torch.exp(0.5j * phase)
                    ), -1)
                )

                state = self.apply_gate(state, ry, wire)
                state = self.apply_gate(state, rz, wire)

            for sign in self.cz_signs:
                state = state * sign

        probabilities = state.abs().square()
        return probabilities @ self.z_diagonals.T
'''
    source = source.replace(anchor, new_class + anchor)

# Na MPC, manter um circuito independente por termo.
old = '''                self.aux = nn.ModuleList([
                    Circuit(i, int(cfg.get("layers", 2)), [word],
                            bool(cfg.get("quantum_trainable", True)))
                    for i, word in enumerate(words, start=1)
                ])'''

new = '''                circuit_class = (
                    ZCircuit if cfg.get("fast_z", False)
                    else Circuit
                )
                self.aux = nn.ModuleList([
                    circuit_class(i, int(cfg.get("layers", 2)), [word],
                                  bool(cfg.get("quantum_trainable", True)))
                    for i, word in enumerate(words, start=1)
                ])'''

if old in source:
    source = source.replace(old, new)
else:
    assert "circuit_class(i," in source, (
        "Construcao da MPC diferente da esperada; nenhuma alteracao salva."
    )

ast.parse(source)
path.write_text(source)

print("Patch instalado.")
print("Backup:", backup)
print("Circuit original preservado como padrao.")
print("ZCircuit ativada apenas com fast_z=True.")
