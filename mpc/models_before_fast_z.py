"""Arquiteturas MLP, MPC classica, MPC hibrida e QNN em PyTorch puro."""
import math
import torch
from torch import nn


def mlp(inputs, outputs, width, depth, activation="tanh"):
    """MLP com tanh por padrao; ultima camada linear."""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(inputs, width), nn.Tanh() if activation == "tanh" else nn.LeakyReLU(0.1)]
        inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


PAULI = {
    "I": torch.eye(2, dtype=torch.complex64),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
}


def pauli_matrix(word, qubits):
    """Observaveis como 'Z', 'XZ', 'ZYX', com uma letra por qubit."""
    if len(word) != qubits or any(p not in PAULI for p in word):
        raise ValueError(f"O observavel '{word}' deve ter {qubits} Pauli(s)")
    operator = PAULI[word[0]]
    for p in word[1:]:
        operator = torch.kron(operator, PAULI[p])
    return operator


class Circuit(nn.Module):
    """Circuito statevector exato e diferenciavel: RY, RZ, CZ e Pauli.

    x=None -> circuito auxiliar: NAO recebe coordenadas; faz uma unica simulacao.
    x=[batch,2] -> QNN convencional: encoding e data-reuploading.
    """
    def __init__(self, qubits, layers, observables, trainable=True):
        super().__init__()
        self.qubits = qubits
        self.layers = layers
        self.theta = nn.Parameter(
            0.4 * torch.randn(layers, qubits, 2), requires_grad=trainable
        )
        self.register_buffer("operators", torch.stack([
            pauli_matrix(word, qubits) for word in observables
        ]))
        pos = torch.arange(2 ** qubits)
        signs = []
        for wire in range(qubits - 1):
            left = (pos >> (qubits - wire - 1)) & 1
            right = (pos >> (qubits - wire - 2)) & 1
            signs.append((1 - 2 * (left * right)).to(torch.complex64))
        self.register_buffer("cz_signs", torch.stack(signs) if signs else
                             torch.empty(0, 2 ** qubits, dtype=torch.complex64))

    def apply_gate(self, state, gate, wire):
        batch = state.shape[0]
        shape = [batch] + [2] * self.qubits
        moved = state.reshape(shape).movedim(wire + 1, -1)
        rotated = torch.bmm(moved.reshape(batch, -1, 2), gate.transpose(1, 2))
        return rotated.reshape(shape).movedim(-1, wire + 1).reshape(batch, -1)

    def forward(self, x=None):
        batch = 1 if x is None else len(x)
        state = torch.zeros(batch, 2 ** self.qubits, dtype=torch.complex64,
                            device=self.theta.device)
        state[:, 0] = 1
        for layer in range(self.layers):
            for wire in range(self.qubits):
                angle = self.theta[layer, wire, 0].expand(batch)
                if x is not None:
                    angle = angle + math.pi * x[:, wire % 2]
                phase = self.theta[layer, wire, 1].expand(batch)
                c, s = torch.cos(angle / 2), torch.sin(angle / 2)
                ry = torch.stack((torch.stack((c, -s), -1),
                                  torch.stack((s, c), -1)), -2).to(torch.complex64)
                rz = torch.diag_embed(torch.stack((torch.exp(-0.5j * phase),
                                                   torch.exp(0.5j * phase)), -1))
                state = self.apply_gate(state, ry, wire)
                state = self.apply_gate(state, rz, wire)
            for sign in self.cz_signs:
                state = state * sign
        return torch.stack([
            (state.conj() * (state @ operator.T)).sum(-1).real
            for operator in self.operators
        ], dim=-1)


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg["kind"]
        self.aggregation = cfg.get("aggregation", "mean")
        self.powers = bool(cfg.get("powers", True))
        width, depth = int(cfg.get("width", 32)), int(cfg.get("depth", 2))
        activation = cfg.get("activation", "tanh")

        if self.kind == "mlp":
            self.main = mlp(2, 1, width, depth, activation)
        elif self.kind in ("mpc_classical", "mpc_hybrid"):
            self.terms = int(cfg.get("terms", 3))
            if self.terms < 1:
                raise ValueError("terms deve ser >= 1")
            if self.aggregation not in ("mean", "sum"):
                raise ValueError("aggregation deve ser mean ou sum")
            self.main = mlp(2, self.terms, width, depth, activation)
            self.register_buffer("exponents", torch.arange(1, self.terms + 1))
            if self.kind == "mpc_classical":
                # A MLP auxiliar recebe APENAS a constante 1, nunca (x,t).
                self.register_buffer("one", torch.ones(1, 1))
                self.aux = mlp(1, self.terms, width, depth, activation)
            else:
                words = cfg.get("observables", ["Z" * i for i in range(1, self.terms + 1)])
                if len(words) != self.terms:
                    raise ValueError("Informe um observavel por termo MPC")
                # Um unico circuito compartilhado por todos os termos.
                qubits = int(cfg.get("qubits", self.terms))
                self.aux = Circuit(
                    qubits,
                    int(cfg.get("layers", 2)),
                    words,
                    bool(cfg.get("quantum_trainable", True))
                )
            # Comeca com termos (1 + y_i) moderados para treinar sem grandes erros.
            nn.init.constant_(self.main[-1].bias, float(cfg.get("main_bias_init", -0.7)))
        elif self.kind == "qnn":
            qubits = int(cfg.get("qubits", 3))
            words = cfg.get("observables", ["Z" + "I" * (qubits - 1)])
            self.circuit = Circuit(qubits, int(cfg.get("layers", 2)), words)
            self.readout = nn.Linear(len(words), 1)
        else:
            raise ValueError(f"Modelo desconhecido: {self.kind}")

    def coefficients(self):
        """Constantes independentes de entrada: z_i ou <P_i>."""
        if self.kind == "mpc_classical":
            return self.aux(self.one).flatten()
        if self.kind == "mpc_hybrid":
            return self.aux().flatten()
        raise ValueError("Somente as MPC possuem coeficientes auxiliares")

    def forward(self, x):
        if self.kind == "mlp":
            return self.main(x)
        if self.kind == "qnn":
            return self.readout(self.circuit(x))
        y = self.main(x)
        z = self.coefficients()  # (terms,) sem dependencia de x ou t
        basis = z.pow(self.exponents) if self.powers else z
        terms = (1 + y) * basis
        return terms.mean(dim=-1, keepdim=True) if self.aggregation == "mean" else terms.sum(dim=-1, keepdim=True)
