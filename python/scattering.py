import logging
import math
import os
from typing import Optional, List

import torch as tr
import torch.nn.functional as F
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T

from scalogram import plot_scalogram
from wavelets import MorletWavelet, make_wavelet_bank

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def scatter(x: T, scat_win: int, dim: int = -1, hop_size: Optional[int] = None) -> T:
    assert x.ndim >= 2
    assert scat_win > 1
    assert x.size(dim) >= scat_win

    if hop_size is None:
        hop_size = scat_win
    else:
        assert hop_size <= scat_win

    unfolded = x.unfold(dimension=dim, size=scat_win, step=hop_size)
    out = tr.mean(unfolded, dim=-1, keepdim=False)
    return out


def calc_scalogram_1d(x: T,
                      wavelet_bank: List[T],
                      scat_win: Optional[int] = None,
                      hop_size: Optional[int] = None) -> T:
    assert x.ndim == 3
    n_ch = x.size(1)

    max_wavelet_len = max([len(w) for w in wavelet_bank])
    max_padding = max_wavelet_len // 2
    # TODO(cm): check why we can get away with only padding the front
    x = F.pad(x, (max_padding, 0))
    x_fd = tr.fft.fft(x).unsqueeze(1)

    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 1
        left_padding = max_padding - wavelet.size(-1) // 2
        right_padding = x_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, -1).expand(-1, n_ch, -1)
        kernel = F.pad(kernel, (left_padding, right_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=0).unsqueeze(0)
    kernels_fd = tr.fft.fft(kernels)
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    y_fd = kernels_fd * x_fd
    y = tr.fft.ifft(y_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    y = y[:, :, :, :-max_padding]
    y = tr.sum(y, dim=2, keepdim=False)
    y = tr.abs(y)

    # TODO(cm): do in the freq domain and add padding for last frame
    if scat_win is not None:
        if scat_win > max_wavelet_len // 6:
            log.warning("Scattering window is suspiciously large (probably greater than the lowest central freq)")
        y = scatter(y, scat_win, hop_size=hop_size)

    return y


def calc_scalogram_2d(x: T,
                      wavelet_bank: List[T],
                      scat_win_f: Optional[int] = None,
                      hop_size_f: Optional[int] = None,
                      scat_win_t: Optional[int] = None,
                      hop_size_t: Optional[int] = None,
                      max_f_dim: Optional[int] = None,
                      max_t_dim: Optional[int] = None) -> T:
    assert x.ndim == 3
    if max_f_dim is None:
        max_f_dim = max([w.size(0) for w in wavelet_bank])
    if max_t_dim is None:
        max_t_dim = max([w.size(1) for w in wavelet_bank])
    max_f_padding = max_f_dim // 2
    max_t_padding = max_t_dim // 2
    # TODO(cm): check why we can get away with only padding the front
    x = F.pad(x, (max_t_padding, 0, max_f_padding, 0))
    log.debug("scalogram fft")
    x_fd = tr.fft.fft2(x).unsqueeze(1)

    log.debug("making kernels")
    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 2
        top_padding = max_f_padding - wavelet.size(-2) // 2
        bottom_padding = x_fd.size(-2) - wavelet.size(-2) - top_padding
        left_padding = max_t_padding - wavelet.size(-1) // 2
        right_padding = x_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, *wavelet.shape)
        kernel = F.pad(kernel, (left_padding, right_padding, top_padding, bottom_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=1)
    log.debug("kernel fft")
    kernels_fd = tr.fft.fft2(kernels)
    log.debug("matmult")
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    y_fd = kernels_fd * x_fd
    log.debug("y ifft")
    y = tr.fft.ifft2(y_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    y = y[:, :, :-max_f_padding, :-max_t_padding]
    y = tr.abs(y)

    # TODO(cm): do in the freq domain and add padding for last frame
    if scat_win_f is not None:
        if scat_win_f > (max_f_dim + 1) // 6:
            log.warning("Freq scattering window is suspiciously large (probably greater than the lowest central freq)")
        y = scatter(y, scat_win_f, dim=-2, hop_size=hop_size_f)

    if scat_win_t is not None:
        if scat_win_t > (max_t_dim + 1) // 6:
            log.warning("Time scattering window is suspiciously large (probably greater than the lowest central freq)")
        y = scatter(y, scat_win_t, dim=-1, hop_size=hop_size_t)

    return y


if __name__ == "__main__":
    # should_scatter = True
    should_scatter = False
    log.info(f"should_scatter = {should_scatter}")

    start_n = 5 * 48000
    n_samples = 24000

    audio_path = "../data/sine_sweep.wav"
    chirp_audio, sr = torchaudio.load(audio_path)
    chirp_audio = chirp_audio[:, start_n:start_n + n_samples]
    chirp_audio = chirp_audio.view(1, 1, -1)

    audio = chirp_audio
    # audio = tr.sin(2 * tr.pi * 220.0 * (1 / sr) * tr.arange(n_samples)).view(1, 1, -1)

    factor = 1
    log.info(f"factor = {factor}")
    sr = sr // factor
    audio = audio[:, :, ::factor]

    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr_t=sr, sr_f=48000)

    J_1 = 5
    Q_1 = 12
    # highest_freq = None
    highest_freq = 6000
    wavelet_bank, freqs = make_wavelet_bank(mw, J_1, Q_1, highest_freq)
    # for w in wavelet_bank:
    #     print(f"{w.shape}")

    lowest_freq = freqs[-1]
    scat_win = None
    # if should_scatter:
    #     assert sr % lowest_freq == 0
    #     scat_win = int(sr / lowest_freq)

    log.info(f"in audio.shape = {audio.shape}")
    log.info(f"scattering_window = {T}")
    scalogram = calc_scalogram_1d(audio, wavelet_bank, scat_win=scat_win)
    log.info(f"scalogram shape = {scalogram.shape}")
    # log.info(f"scalogram mean = {tr.mean(scalogram)}")
    # log.info(f"scalogram std = {tr.std(scalogram)}")
    # log.info(f"scalogram max = {tr.max(scalogram)}")
    # log.info(f"scalogram min = {tr.min(scalogram)}")
    log.info(f"scalogram energy = {MorletWavelet.calc_energy(scalogram)}")
    scalogram *= 2 ** (math.log2(factor))
    log.info(f"scalogram energy fixed = {MorletWavelet.calc_energy(scalogram)}")
    # plot_scalogram(scalogram[0], title="chirp", dt=mw.dt, freqs=freqs)
    # exit()

    J_2_f = 2
    Q_2_f = 1
    # highest_freq_f = None
    highest_freq_f = 1500
    J_2_t = 3
    Q_2_t = 1
    # highest_freq_t = None
    highest_freq_t = 6000

    wavelet_bank_2, freqs_2 = make_wavelet_bank(mw, J_2_t, Q_2_t, highest_freq_t, J_2_f, Q_2_f, highest_freq_f)
    # for w in wavelet_bank_2:
    #     print(f"{w.shape}")

    lowest_freq_f, lowest_freq_t, _ = freqs_2[-1]
    scat_win_f = None
    scat_win_t = None
    if should_scatter:
        assert sr % lowest_freq_f == 0
        scat_win_f = int(sr / lowest_freq_f)
        assert sr % lowest_freq_t == 0
        scat_win_t = int(sr / lowest_freq_t)

    jtfst = calc_scalogram_2d(scalogram, wavelet_bank_2, scat_win_f=scat_win_f, scat_win_t=scat_win_t)
    log.info(f"jtfst shape = {jtfst.shape}")
    # log.info(f"jtfst mean = {tr.mean(jtfst)}")
    # log.info(f"jtfst std = {tr.std(jtfst)}")
    # log.info(f"jtfst max = {tr.max(jtfst)}")
    # log.info(f"jtfst min = {tr.min(jtfst)}")
    log.info(f"jtfst energy = {MorletWavelet.calc_energy(jtfst)}")
    jtfst *= 2 ** (math.log2(factor) / 2)
    log.info(f"jtfst energy fixed = {MorletWavelet.calc_energy(jtfst)}")
    for idx, w in enumerate(wavelet_bank_2):
        log.info(f"{idx}: {MorletWavelet.calc_energy(jtfst[0, idx, :, :]):.2f}, shape = {w.shape}")

    # pic = jtfst[0, 0, :, :].squeeze().detach().numpy()
    # plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    # plt.show()
    # exit()

    n_rows = len(freqs_2) // 2
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx, (freq_f, freq_t, theta) in enumerate(freqs_2):
        curr_ax = ax[idx // 2, idx % 2]
        pic = jtfst[0, idx, :, :].squeeze().detach().numpy()
        curr_ax.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
        curr_ax.set_title(f"freq_f = {freq_f:.0f}, freq_t = {freq_t:.0f}, theta = {theta}")
    plt.show()
