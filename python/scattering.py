import logging
import os
import time
from typing import Optional, List, Tuple
import torch.nn.functional as F
import numpy as np
import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T
from tqdm import tqdm

from filterbanks import make_wavelet_bank
from scalogram_1d import calc_scalogram_1d, plot_scalogram_1d
from scalogram_2d import calc_scalogram_2d
from signals import make_pure_sine, make_pulse, make_exp_chirp, make_hann_window
from wavelets import MorletWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def average_td(x: T, avg_win: int, dim: int = -1, hop_size: Optional[int] = None) -> T:
    assert x.ndim >= 2
    assert avg_win >= 1
    assert x.size(dim) >= avg_win

    if hop_size is None:
        hop_size = avg_win

    # TODO(cm): add padding for last frame
    unfolded = x.unfold(dimension=dim, size=avg_win, step=hop_size)
    out = tr.mean(unfolded, dim=-1, keepdim=False)
    return out


def _calc_scat_transform_1d(x: T,
                            sr: float,
                            wavelet_bank: List[T],
                            freqs: List[float],
                            should_avg: bool = False,
                            avg_win: Optional[int] = None,
                            squeeze_channels: bool = True) -> T:
    assert x.ndim == 3
    assert len(wavelet_bank) == len(freqs)
    y = calc_scalogram_1d(x, wavelet_bank, take_modulus=True, squeeze_channels=squeeze_channels)

    if not should_avg:
        return y

    if avg_win is None:
        lowest_freq = freqs[-1]
        assert sr % lowest_freq == 0
        avg_win = int(sr / lowest_freq)

    log.debug(f"avg_win = {avg_win}")
    max_wavelet_len = max([w.size(0) for w in wavelet_bank])
    if avg_win > (max_wavelet_len + 1) // 6:
        log.warning("Averaging window is suspiciously large (probably greater than the lowest central freq)")
    y = average_td(y, avg_win, dim=-1)
    return y


def calc_scat_transform_1d(x: T,
                           sr: float,
                           J: int,
                           Q: int = 1,
                           should_avg: bool = False,
                           highest_freq: Optional[float] = None,
                           avg_win: Optional[int] = None) -> (T, List[float], List[T]):
    assert x.ndim == 3
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr_t=sr)
    wavelet_bank, freqs = make_wavelet_bank(mw, J, Q, highest_freq)
    y = _calc_scat_transform_1d(x, sr, wavelet_bank, freqs, should_avg, avg_win)
    return y, freqs, wavelet_bank


def calc_scat_transform_1d_fast(x: T,
                                sr: float,
                                J: int,
                                Q: int = 1) -> (T, List[float]):
    assert x.ndim == 3
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    curr_x = x
    curr_sr = sr
    curr_avg_win = 2 ** (J + 1)
    octaves = []
    freqs_all = []
    for curr_j in tqdm(range(J)):
        if curr_j == J - 1:
            include_lowest_octave = True
        else:
            include_lowest_octave = False

        mw = MorletWavelet(w=w, sr_t=curr_sr)
        wavelet_bank, freqs = make_wavelet_bank(
            mw, n_octaves_t=1, steps_per_octave_t=Q, include_lowest_octave_t=include_lowest_octave)
        octave = calc_scalogram_1d(curr_x, wavelet_bank, take_modulus=True)
        octave = average_td(octave, curr_avg_win)
        octaves.append(octave)
        freqs_all.extend(freqs)
        curr_x = average_td(curr_x, avg_win=2, hop_size=2)
        curr_sr /= 2
        curr_avg_win //= 2
    y = tr.cat(octaves, dim=-2)
    return y, freqs_all


def calc_scat_transform_1d_jagged(x: T,
                                  sr: float,
                                  freqs_x: List[float],
                                  J: int,
                                  Q: int = 1,
                                  should_avg: bool = False,
                                  highest_freq: Optional[float] = None,
                                  avg_win: Optional[int] = None,
                                  should_pad: bool = False) -> (T, List[float]):
    assert x.ndim == 3
    assert x.size(1) == len(freqs_x)
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr_t=sr)
    wavelet_bank, freqs = make_wavelet_bank(mw, J, Q, highest_freq)
    y_s = []
    freqs_out = []
    for wavelet, freq in tqdm(zip(wavelet_bank, freqs)):
        band_freqs = [f_x for f_x in freqs_x if f_x >= 2 * freq]
        n_bands = len(band_freqs)
        if n_bands == 0:
            continue
        curr_x = x[:, :n_bands, :]
        curr_wb = [wavelet]
        curr_freqs = [freq]
        y = _calc_scat_transform_1d(curr_x, sr, curr_wb, curr_freqs, should_avg, avg_win, squeeze_channels=False)
        if should_pad:
            pad_n = x.size(1) - n_bands
            y = F.pad(y, (0, 0, 0, pad_n))
        y_s.append(y)
        plt.imshow(y.squeeze(0).squeeze(0).numpy(), aspect="auto")
        plt.show()
        freqs_out.append(freq)

    return y_s, freqs_out


def calc_scat_transform_2d(x: T,
                           sr: float,
                           J_f: int,
                           J_t: int,
                           Q_f: int = 1,
                           Q_t: int = 1,
                           should_scatter_f: bool = False,
                           should_scatter_t: bool = True,
                           highest_freq_f: Optional[float] = None,
                           highest_freq_t: Optional[float] = None,
                           scat_win_f: Optional[int] = None,
                           scat_win_t: Optional[int] = None,
                           hop_size_f: Optional[int] = None,
                           hop_size_t: Optional[int] = None) -> (T, List[Tuple[float, float, int]], List[T]):
    assert x.ndim == 3
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr_t=sr, sr_f=48000)
    wavelet_bank, freqs = make_wavelet_bank(mw,
                                            J_t,
                                            Q_t,
                                            highest_freq_t,
                                            n_octaves_f=J_f,
                                            steps_per_octave_f=Q_f,
                                            highest_freq_f=highest_freq_f)
    y = calc_scalogram_2d(x, wavelet_bank, take_modulus=True)

    if not should_scatter_f and not should_scatter_t:
        return y, freqs, wavelet_bank

    lowest_freq_f, lowest_freq_t, _ = freqs[-1]
    if should_scatter_t:
        if scat_win_t is None:
            assert mw.sr_t % lowest_freq_t == 0
            scat_win_t = int(mw.sr_t / lowest_freq_t)

        log.info(f"scat_win_t = {scat_win_t}")
        max_wavelet_len_t = max([w.size(1) for w in wavelet_bank])
        if scat_win_t > (max_wavelet_len_t + 1) // 6:
            log.warning("Time scattering window is suspiciously large (probably greater than the lowest central freq)")
        y = average_td(y, scat_win_t, dim=-1, hop_size=hop_size_t)

    if should_scatter_f:
        if scat_win_f is None:
            log.warning(f"should_scatter_f is True, but scat_win_f is None, using a heuristic value of 2")
            scat_win_f = 2

        log.info(f"scat_win_f = {scat_win_f}")
        max_wavelet_len_f = max([w.size(0) for w in wavelet_bank])
        if scat_win_f > (max_wavelet_len_f + 1) // 6:
            log.warning("Freq scattering window is suspiciously large (probably greater than the lowest central freq)")
        y = average_td(y, scat_win_f, dim=-2, hop_size=hop_size_f)

    return y, freqs, wavelet_bank


def calc_scat_transform_2d_fast(x: T,
                                sr: float,
                                J_f: int,
                                J_t: int,
                                Q_f: int = 1,
                                Q_t: int = 1,
                                should_scatter_f: bool = False,
                                should_scatter_t: bool = True,
                                scat_win_f: Optional[int] = None,
                                scat_win_t: Optional[int] = None) -> (T, List[Tuple[float, float, int]]):
    if should_scatter_f:
        if scat_win_f is None:
            log.warning(f"should_scatter_f is True, but scat_win_f is None, using a heuristic value of 2")
            scat_win_f = 2
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    curr_x = x
    curr_sr_t = sr
    curr_highest_freq_t = sr / 2

    if should_scatter_t:
        if scat_win_t is None:
            curr_scat_win_t = 2 ** J_t
        else:
            curr_scat_win_t = scat_win_t
    else:
        curr_scat_win_t = 1

    octaves = []
    freqs_all = []
    for curr_j_t in tqdm(range(J_t)):
        if curr_j_t == J_t - 1:
            include_lowest_octave_t = True
        else:
            include_lowest_octave_t = False

        mw = MorletWavelet(w=w, sr_t=curr_sr_t, sr_f=sr)
        wavelet_bank, freqs = make_wavelet_bank(mw,
                                                n_octaves_t=1,
                                                steps_per_octave_t=Q_t,
                                                highest_freq_t=curr_highest_freq_t,
                                                include_lowest_octave_t=include_lowest_octave_t,
                                                n_octaves_f=J_f,
                                                steps_per_octave_f=Q_f)
        octave = calc_scalogram_2d(curr_x, wavelet_bank, take_modulus=True)
        octave = average_td(octave, curr_scat_win_t, dim=-1)
        if should_scatter_f:
            octave = average_td(octave, scat_win_f, dim=-2)
        octaves.append(octave)
        freqs_all.extend(freqs)
        if curr_scat_win_t > 1:
            curr_x = average_td(curr_x, avg_win=2, dim=-1, hop_size=2)
            curr_sr_t /= 2
            assert curr_scat_win_t % 2 == 0
            curr_scat_win_t //= 2
        else:
            log.info(f"stopped subsampling time at freq {curr_highest_freq_t}")
        curr_highest_freq_t /= 2
    y = tr.cat(octaves, dim=-3)
    return y, freqs_all


if __name__ == "__main__":
    should_avg_1 = False
    should_avg_f = False
    should_avg_t = False
    log.info(f"should_avg_1 = {should_avg_1}")
    log.info(f"should_avg_f = {should_avg_f}")
    log.info(f"should_avg_t = {should_avg_t}")

    # start_n = int(4 * 48000)
    # n_samples = 2 * 48000
    start_n = 0
    n_samples = 2 ** 16

    audio_path = "../data/sine_sweep.wav"
    chirp_audio, sr = torchaudio.load(audio_path)
    chirp_audio = chirp_audio[:, start_n:start_n + n_samples]
    chirp_audio = chirp_audio.view(1, 1, -1)

    audio_path = "../data/flute.wav"
    flute_audio, sr = torchaudio.load(audio_path)
    flute_audio = flute_audio[:, start_n:n_samples]
    flute_audio = tr.mean(flute_audio, dim=0)
    flute_audio = flute_audio.view(1, 1, -1)

    # audio = chirp_audio
    audio = flute_audio
    sr = 48000
    n_samples = 4096

    audio_1 = make_pure_sine(n_samples, sr, freq=4000, amp=1.0)
    audio_2 = make_pulse(n_samples, center_loc=0.5, dur_samples=16, amp=4.0)
    audio_3 = make_exp_chirp(n_samples, sr, start_freq=750, end_freq=20000, amp=1.0)
    audio = audio_1 + audio_2 + audio_3

    audio = audio.view(1, 1, -1)

    factor = 1
    log.info(f"factor = {factor}")
    sr = sr // factor
    audio = audio[:, :, ::factor]

    J_1 = 7
    Q_1 = 16
    highest_freq = None
    # highest_freq = 6000

    log.info(f"in audio.shape = {audio.shape}")
    scalogram, freqs, wavelet_bank = calc_scat_transform_1d(audio,
                                                            sr,
                                                            J_1,
                                                            Q_1,
                                                            should_avg_1,
                                                            highest_freq=highest_freq,
                                                            avg_win=None)

    log.info(f"scalogram shape = {scalogram.shape}")
    # log.info(f"scalogram mean = {tr.mean(scalogram)}")
    # log.info(f"scalogram std = {tr.std(scalogram)}")
    # log.info(f"scalogram max = {tr.max(scalogram)}")
    # log.info(f"scalogram min = {tr.min(scalogram)}")
    log.info(f"scalogram energy = {MorletWavelet.calc_energy(scalogram)}")

    mean = tr.mean(scalogram)
    std = tr.std(scalogram)
    scalogram_to_plot = tr.clip(scalogram, mean - (4 * std), mean + (4 * std))

    plot_scalogram_1d(scalogram_to_plot[0], title="scalo", dt=1.0 / sr, freqs=freqs, n_y_ticks=J_1)

    # scalogram_fast, freqs_fast = calc_scat_transform_1d_fast(audio, sr, J_1, Q_1)
    # log.info(f"scalogram_fast shape = {scalogram_fast.shape}")
    # # log.info(f"scalogram_fast mean = {tr.mean(scalogram_fast)}")
    # # log.info(f"scalogram_fast std = {tr.std(scalogram_fast)}")
    # # log.info(f"scalogram_fast max = {tr.max(scalogram_fast)}")
    # # log.info(f"scalogram_fast min = {tr.min(scalogram_fast)}")
    # log.info(f"scalogram_fast energy = {MorletWavelet.calc_energy(scalogram_fast)}")
    # plot_scalogram_1d(scalogram_fast[0], title="scalo fast", dt=1.0 / sr, freqs=freqs_fast)
    exit()

    J_2_f = 6
    Q_2_f = 2
    # highest_freq_f = None
    highest_freq_f = 6000
    J_2_t = 3
    Q_2_t = 1
    # highest_freq_t = None
    highest_freq_t = 600
    avg_win_f = None
    # avg_win_f = 2 ** 2
    avg_win_t = None
    # avg_win_t = 2 ** 11

    # y_2, freqs_2 = calc_scat_transform_1d_jagged(scalogram,
    #                                              sr,
    #                                              freqs,
    #                                              J_2_t,
    #                                              Q_2_t,
    #                                              should_avg=should_scatter_t,
    #                                              highest_freq=highest_freq_t,
    #                                              avg_win=avg_win_t,
    #                                              should_pad=True)
    # exit()

    pic_idx = 12

    start_t = time.perf_counter()
    jtfst, freqs_2, wavelet_bank_2 = calc_scat_transform_2d(scalogram,
                                                            sr,
                                                            J_2_f,
                                                            J_2_t,
                                                            Q_2_f,
                                                            Q_2_t,
                                                            should_avg_f,
                                                            should_avg_t,
                                                            highest_freq_f,
                                                            highest_freq_t,
                                                            avg_win_f,
                                                            avg_win_t)
    end_t = time.perf_counter()
    log.info(f"elapsed time = {end_t - start_t:.2f} seconds")
    log.info(f"jtfst shape = {jtfst.shape}")
    # log.info(f"jtfst mean = {tr.mean(jtfst)}")
    # log.info(f"jtfst std = {tr.std(jtfst)}")
    # log.info(f"jtfst max = {tr.max(jtfst)}")
    # log.info(f"jtfst min = {tr.min(jtfst)}")
    log.info(f"jtfst energy = {MorletWavelet.calc_energy(jtfst)}")

    mean = tr.mean(jtfst)
    std = tr.std(jtfst)
    jtfst = tr.clip(jtfst, mean - (4 * std), mean + (4 * std))

    pic = jtfst[0, pic_idx, :, :].squeeze().detach().numpy()
    plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    plt.title("jtfst")
    plt.show()

    start_t = time.perf_counter()
    jtfst_fast, freqs_2_fast = calc_scat_transform_2d_fast(scalogram,
                                                           sr,
                                                           J_2_f,
                                                           J_2_t,
                                                           Q_2_f,
                                                           Q_2_t,
                                                           should_avg_f,
                                                           should_avg_t,
                                                           avg_win_f,
                                                           avg_win_t)
    end_t = time.perf_counter()
    log.info(f"elapsed time = {end_t - start_t:.2f} seconds")
    log.info(f"lowest_freq_t = {freqs_2_fast[-1][1]:.2f}")
    log.info(f"jtfst_fast shape = {jtfst_fast.shape}")
    # log.info(f"jtfst_fast mean = {tr.mean(jtfst_fast)}")
    # log.info(f"jtfst_fast std = {tr.std(jtfst_fast)}")
    # log.info(f"jtfst_fast max = {tr.max(jtfst_fast)}")
    # log.info(f"jtfst_fast min = {tr.min(jtfst_fast)}")
    log.info(f"jtfst_fast energy = {MorletWavelet.calc_energy(jtfst_fast)}")

    mean = tr.mean(jtfst_fast)
    std = tr.std(jtfst_fast)
    jtfst_fast = tr.clip(jtfst_fast, mean - (4 * std), mean + (4 * std))

    pic = jtfst_fast[0, pic_idx, :, :].squeeze().detach().numpy()
    plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    plt.title("jtfst_fast")
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
