import logging
import os
from typing import Optional, List

import matplotlib.pyplot as plt
import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from tqdm import tqdm

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


def calc_scalogram_td(audio: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert audio.ndim == 3
    assert wavelet_bank
    n_ch = audio.size(1)

    audio_complex = audio.to(wavelet_bank[0].dtype)
    convs = []
    for wavelet in tqdm(wavelet_bank):
        assert wavelet.ndim == 1
        kernel = wavelet.view(1, 1, -1).repeat(1, n_ch, 1)
        out = F.conv1d(audio_complex, kernel, stride=(1,), padding="same")
        convs.append(out)

    scalogram = tr.cat(convs, dim=1)
    if take_modulus:
        scalogram = tr.abs(scalogram)

    return scalogram


def calc_scalogram_fd(audio: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert audio.ndim == 3
    n_ch = audio.size(1)

    max_wavelet_len = max([len(w) for w in wavelet_bank])
    max_padding = max_wavelet_len // 2
    # TODO(cm): check why we can get away with only padding the front
    audio = F.pad(audio, (max_padding, 0))
    audio_fd = tr.fft.fft(audio).unsqueeze(1)

    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 1
        left_padding = max_padding - wavelet.size(-1) // 2
        right_padding = audio_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, -1).expand(-1, n_ch, -1)
        kernel = F.pad(kernel, (left_padding, right_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=0).unsqueeze(0)
    kernels_fd = tr.fft.fft(kernels)
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    out_fd = kernels_fd * audio_fd
    scalogram = tr.fft.ifft(out_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    scalogram = scalogram[:, :, :, :-max_padding]
    scalogram = tr.sum(scalogram, dim=2, keepdim=False)

    if take_modulus:
        scalogram = tr.abs(scalogram)

    return scalogram
