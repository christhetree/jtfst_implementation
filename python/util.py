import logging
import os
from typing import Optional, List

import matplotlib.pyplot as plt
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def plot_scalogram(scalogram: T,
                   dt: Optional[float] = None,
                   freqs: Optional[List[float]] = None,
                   title: str = "scalogram",
                   n_x_ticks: int = 8,
                   n_y_ticks: int = 8,
                   interpolation: str = "none",
                   cmap: str = "OrRd") -> None:
    assert scalogram.ndim == 2
    scalogram = scalogram.detach().numpy()
    plt.imshow(scalogram, aspect="auto", interpolation=interpolation, cmap=cmap)
    plt.title(title)

    if dt:
        x_pos = list(range(scalogram.shape[1]))
        x_labels = [pos * dt for pos in x_pos]
        x_step_size = len(x_pos) // n_x_ticks
        x_pos = x_pos[::x_step_size]
        x_labels = x_labels[::x_step_size]
        x_labels = [f"{_:.3f}" for _ in x_labels]
        plt.xticks(x_pos, x_labels)
        plt.xlabel("time (s)")
    else:
        plt.xlabel("samples")

    if freqs is not None:
        assert scalogram.shape[0] == len(freqs)
        y_pos = list(range(len(freqs)))
        y_step_size = len(freqs) // n_y_ticks
        y_pos = y_pos[::y_step_size]
        y_labels = freqs[::y_step_size]
        y_labels = [f"{_:.0f}" for _ in y_labels]
        plt.yticks(y_pos, y_labels)
        plt.ylabel("freq (Hz)")

    plt.show()
