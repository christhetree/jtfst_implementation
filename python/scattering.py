import logging
import math
import os
from typing import Optional, List, Tuple

import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T

from filterbanks import make_wavelet_bank
from scalogram_1d import calc_scalogram_1d, plot_scalogram_1d
from scalogram_2d import calc_scalogram_2d
from wavelets import MorletWavelet

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

    # TODO(cm): do in the freq domain and add padding for last frame
    unfolded = x.unfold(dimension=dim, size=scat_win, step=hop_size)
    out = tr.mean(unfolded, dim=-1, keepdim=False)
    return out


def calc_scat_transform_1d(x: T,
                           sr: float,
                           J: int,
                           Q: int = 1,
                           should_scatter: bool = True,
                           highest_freq: Optional[float] = None,
                           scat_win: Optional[int] = None,
                           hop_size: Optional[int] = None) -> (T, List[float], List[T]):
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr_t=sr)
    wavelet_bank, freqs = make_wavelet_bank(mw, J, Q, highest_freq)
    y = calc_scalogram_1d(x, wavelet_bank, take_modulus=True)

    if not should_scatter:
        return y, freqs, wavelet_bank

    if scat_win is None:
        lowest_freq = freqs[-1]
        log.info(f"lowest_freq = {lowest_freq}")
        assert sr % lowest_freq == 0
        scat_win = int(sr / lowest_freq)

    log.info(f"scat_win = {scat_win}")
    max_wavelet_len = max([len(w) for w in wavelet_bank])
    if scat_win > max_wavelet_len + 1 // 6:
        log.warning("Scattering window is suspiciously large (probably greater than the lowest central freq)")
    y = scatter(y, scat_win, dim=-1, hop_size=hop_size)
    return y, freqs, wavelet_bank


def calc_scat_transform_2d(x: T,
                           sr: float,
                           J_f: int,
                           J_t: int,
                           Q_f: int = 1,
                           Q_t: int = 1,
                           should_scatter_f: bool = True,
                           should_scatter_t: bool = False,
                           highest_freq_f: Optional[float] = None,
                           highest_freq_t: Optional[float] = None,
                           scat_win_f: Optional[int] = None,
                           scat_win_t: Optional[int] = None,
                           hop_size_f: Optional[int] = None,
                           hop_size_t: Optional[int] = None) -> (T, List[Tuple[float, float, int]], List[T]):
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr_t=sr)
    wavelet_bank, freqs = make_wavelet_bank(mw, J_t, Q_t, highest_freq_t, J_f, Q_f, highest_freq_f)
    y = calc_scalogram_2d(x, wavelet_bank, take_modulus=True)

    if not should_scatter_f and not should_scatter_t:
        return y, freqs, wavelet_bank

    lowest_freq_f, lowest_freq_t, _ = freqs[-1]
    if should_scatter_t:
        if scat_win_t is None:
            log.info(f"lowest_freq_t = {lowest_freq_t}")
            assert mw.sr_t % lowest_freq_t == 0
            scat_win_t = int(mw.sr_t / lowest_freq_t)

        log.info(f"scat_win_t = {scat_win_t}")
        max_wavelet_len_t = max([w.size(1) for w in wavelet_bank])
        if scat_win_t > max_wavelet_len_t + 1 // 6:
            log.warning("Time scattering window is suspiciously large (probably greater than the lowest central freq)")
        y = scatter(y, scat_win_t, dim=-1, hop_size=hop_size_t)

    if should_scatter_f:
        if scat_win_f is None:
            log.warning(f"should_scatter_f is True, but scat_win_f is None, using a heuristic value of 2")
            scat_win_f = 2

        log.info(f"scat_win_f = {scat_win_f}")
        max_wavelet_len_f = max([w.size(0) for w in wavelet_bank])
        if scat_win_f > max_wavelet_len_f + 1 // 6:
            log.warning("Freq scattering window is suspiciously large (probably greater than the lowest central freq)")
        y = scatter(y, scat_win_f, dim=-2, hop_size=hop_size_f)

    return y, freqs, wavelet_bank


if __name__ == "__main__":
    should_scatter_1 = False
    should_scatter_f = False
    should_scatter_t = False
    log.info(f"should_scatter_1 = {should_scatter_1}")
    log.info(f"should_scatter_f = {should_scatter_f}")
    log.info(f"should_scatter_t = {should_scatter_t}")

    start_n = int(5 * 48000)
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

    J_1 = 4
    Q_1 = 12
    # highest_freq = None
    highest_freq = 6000

    log.info(f"in audio.shape = {audio.shape}")
    scalogram, freqs, wavelet_bank = calc_scat_transform_1d(
        audio, sr, J_1, Q_1, should_scatter_1, highest_freq=highest_freq)

    log.info(f"scalogram shape = {scalogram.shape}")
    # log.info(f"scalogram mean = {tr.mean(scalogram)}")
    # log.info(f"scalogram std = {tr.std(scalogram)}")
    # log.info(f"scalogram max = {tr.max(scalogram)}")
    # log.info(f"scalogram min = {tr.min(scalogram)}")
    log.info(f"scalogram energy = {MorletWavelet.calc_energy(scalogram)}")
    scalogram *= 2 ** (math.log2(factor))
    log.info(f"scalogram energy fixed = {MorletWavelet.calc_energy(scalogram)}")
    plot_scalogram_1d(scalogram[0], title="chirp", dt=1.0 / sr, freqs=freqs)
    # exit()

    J_2_f = 3
    Q_2_f = 1
    # highest_freq_f = None
    highest_freq_f = 6000
    J_2_t = 3
    Q_2_t = 1
    # highest_freq_t = None
    highest_freq_t = 6000

    jtfst, freqs_2, wavelet_bank_2 = calc_scat_transform_2d(
        scalogram, sr, J_2_f, J_2_t, Q_2_f, Q_2_t, should_scatter_f, should_scatter_t, highest_freq_f, highest_freq_t)
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

    pic = jtfst[0, 0, :, :].squeeze().detach().numpy()
    plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    plt.show()
    exit()

    n_rows = len(freqs_2) // 2
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx, (freq_f, freq_t, theta) in enumerate(freqs_2):
        curr_ax = ax[idx // 2, idx % 2]
        pic = jtfst[0, idx, :, :].squeeze().detach().numpy()
        curr_ax.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
        curr_ax.set_title(f"freq_f = {freq_f:.0f}, freq_t = {freq_t:.0f}, theta = {theta}")
    plt.show()
