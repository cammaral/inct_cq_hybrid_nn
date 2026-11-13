from pathlib import Path
import ast
import shutil

path = Path("mpc/models.py")
source = path.read_text()

if "class ZCircuit(Circuit):" in source:
    print("ZCircuit ja instalada.")
    raise SystemExit(0)

anchor = "\n\nclass Model(nn.Module):"
assert source.count(anchor) == 1

new_class = '''

class ZCircuit(Circuit):
    """Circuito auxiliar otimizado para observaveis Z locais.

    Mantem o ansatz RY-RZ-CZ original.
    Nao constroi matrizes densas dos observaveis.
    Retorna somente os valores esperados solicitados.
    """

    def __init__(self, qubits, layers, observables, trainable=True):
        nn.Module.__init__(self)

        self.qubits = qubits
        self.layers = layers

        self.theta = nn.Parameter(
            0.4 * torch.randn(layers, qubits, 2),
            requires_grad=trainable
        )

        # Aceitar apenas observaveis Z_i, um Z por string.
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
        # Esta versao e apenas para o circuito auxiliar,
        # que nao recebe as coordenadas (x,t).
        assert x is None

        batch = 1

        state = torch.zeros(
            batch, 2 ** self.qubits,
            dtype=torch.complex64,
            device=self.theta.device
        )
        state[:, 0] = 1

        for layer in range(self.layers):
            for wire in range(self.qubits):
                angle = self.theta[layer, wire, 0].expand(batch)
                phase = self.theta[layer, wire, 1].expand(batch)

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

        # Somente probabilidades para calcular <Z_i>.
        probabilities = state.abs().square()

        return probabilities @ self.z_diagonals.T
'''

source = source.replace(anchor, new_class + anchor)

old = '''                self.aux = Circuit(
                    qubits,
                    int(cfg.get("layers", 2)),
                    words,
                    bool(cfg.get("quantum_trainable", True))
                )'''

new = '''                circuit_class = (
                    ZCircuit if cfg.get("fast_z", False)
                    else Circuit
                )
                self.aux = circuit_class(
                    qubits,
                    int(cfg.get("layers", 2)),
                    words,
                    bool(cfg.get("quantum_trainable", True))
                )'''

assert source.count(old) == 1
source = source.replace(old, new)

ast.parse(source)

backup = path.with_name("models_before_fast_z.py")
shutil.copy2(path, backup)
path.write_text(source)

print("ZCircuit instalada com sucesso.")
print("Backup:", backup)
print("Modelos antigos: comportamento preservado.")
