from pathlib import Path

path = Path("mpc/train.py")
source = path.read_text()

replacements = [
    (
        '    history = []\n',
        '''    history_path = folder / "history.csv"
    history_file = history_path.open("w", newline="", buffering=1)
    history_writer = csv.writer(history_file)
    history_writer.writerow([
        "epoch", "train_mse", "val_mse",
        "grad_all", "grad_aux", "lr"
    ])
'''
    ),
    (
        '''        history.append([epoch, loss_sum / len(x), val_mse,
                        grad_sum / steps, aux_grad_sum / steps,
                        optimizer.param_groups[0]["lr"]])''',
        '''        history_writer.writerow([
            epoch, loss_sum / len(x), val_mse,
            grad_sum / steps, aux_grad_sum / steps,
            optimizer.param_groups[0]["lr"]
        ])'''
    ),
    (
        '''    net.load_state_dict(best_weights)  # modelo selecionado pelo conjunto de validacao''',
        '''    history_file.close()
    torch.save(net.state_dict(), folder / "final.pt")
    net.load_state_dict(best_weights)  # modelo selecionado pelo conjunto de validacao'''
    ),
    (
        '''    csv_file(folder / "history.csv",
             ["epoch", "train_mse", "val_mse", "grad_all", "grad_aux", "lr"], history)
    plot_history(folder / "loss.png", history)''',
        '''    with history_path.open(newline="") as f:
        history = [
            [float(v) for v in row]
            for row in list(csv.reader(f))[1:]
        ]
    plot_history(folder / "loss.png", history)'''
    ),
]

for old, new in replacements:
    if source.count(old) != 1:
        raise RuntimeError(
            "Trecho nao encontrado ou duplicado; nenhuma alteracao aplicada."
        )

for old, new in replacements:
    source = source.replace(old, new, 1)

path.write_text(source)
print("Salvamento por epoca atualizado.")
