from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .config import TrainConfig


def moving_average(x, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training_curves(
    train_cfg: TrainConfig,
    window: int = 50,
    show: bool = False,
    output_path: Optional[str] = None,
) -> None:
    returns_path = os.path.join(train_cfg.result_dir, "returns.npy")
    losses_path = os.path.join(train_cfg.result_dir, "losses.npy")

    if not os.path.exists(returns_path):
        print(f"returns file not found: {returns_path}")
        return

    returns = np.load(returns_path)
    losses = np.load(losses_path) if os.path.exists(losses_path) else None

    ma_returns = moving_average(returns, window)

    plt.figure()
    plt.plot(returns, alpha=0.3, label="Episode return")
    plt.plot(
        range(window - 1, window - 1 + len(ma_returns)),
        ma_returns,
        label=f"Moving average (window={window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.title("Training Returns")

    if output_path is None:
        output_path = os.path.join(train_cfg.result_dir, "returns_plot.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved return plot to {output_path}")

    if show:
        plt.show()
    plt.close()

    if losses is not None and len(losses) > 0:
        plt.figure()
        plt.plot(losses, alpha=0.7, label="DQN loss")
        plt.xlabel("Update step (approx)")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")

        loss_path = os.path.join(train_cfg.result_dir, "losses_plot.png")
        plt.savefig(loss_path, bbox_inches="tight")
        print(f"Saved loss plot to {loss_path}")

        if show:
            plt.show()
        plt.close()