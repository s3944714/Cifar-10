"""Plotting and experiment logging helpers, reused by every notebook."""
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history, title, save_path=None):
    """Plot training vs validation loss and accuracy side by side."""
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(h["loss"], label="train")
    ax1.plot(h["val_loss"], label="validation")
    ax1.set(title=f"{title}: loss", xlabel="epoch", ylabel="cross-entropy")
    ax1.legend()

    ax2.plot(h["accuracy"], label="train")
    ax2.plot(h["val_accuracy"], label="validation")
    ax2.set(title=f"{title}: accuracy", xlabel="epoch", ylabel="accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)   # save BEFORE show, which clears the figure
    plt.show()


def log_experiment(name, model, history, test_acc=None, notes="",
                   path="../results/experiments.csv"):
    """Append one row per training run, so results are never lost."""
    h = history.history
    row = {
        "date": datetime.date.today().isoformat(),
        "model": name,
        "params": model.count_params(),
        "epochs": len(h["loss"]),
        "best_val_acc": round(max(h["val_accuracy"]), 4),
        "test_acc": None if test_acc is None else round(test_acc, 4),
        "notes": notes,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # header only on the first write; mode="a" appends after that
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return row