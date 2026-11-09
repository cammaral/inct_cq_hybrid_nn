from pathlib import Path

path = Path("mpc/train.py")
s = path.read_text()

def trocar(antigo, novo):
    global s
    assert s.count(antigo) == 1, f"Trecho nao encontrado ou duplicado:\n{antigo}"
    s = s.replace(antigo, novo)

trocar(
    '"grad_all", "grad_aux", "lr"',
    '"grad_all", "grad_aux", "grad_classical", "grad_quantum", "lr"'
)

trocar(
    'loss_sum = grad_sum = aux_grad_sum = 0.0',
    'loss_sum = grad_sum = aux_grad_sum = classical_grad_sum = quantum_grad_sum = 0.0'
)

trocar(
    '''grad_sum += grad_norm(net.parameters())
            if is_mpc:
                aux_grad_sum += grad_norm(net.aux.parameters())''',
    '''grad_sum += grad_norm(net.parameters())
            if hasattr(net, "main"):
                classical_grad_sum += grad_norm(net.main.parameters())
            if is_mpc:
                aux_grad_sum += grad_norm(net.aux.parameters())
                if variant["kind"] == "mpc_hybrid":
                    quantum_grad_sum += grad_norm(net.aux.parameters())'''
)

trocar(
    '''grad_sum / steps, aux_grad_sum / steps,
            optimizer.param_groups[0]["lr"]''',
    '''grad_sum / steps, aux_grad_sum / steps,
            classical_grad_sum / steps, quantum_grad_sum / steps,
            optimizer.param_groups[0]["lr"]'''
)

path.write_text(s)
print("Gradientes classicos e quanticos adicionados.")
