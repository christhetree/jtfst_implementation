import logging
import os
from typing import Optional, List, Union

import torch as tr
import torch.nn.functional as F
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T, nn
from tqdm import tqdm

from filterbanks import make_wavelet_bank
from scalogram_1d import calc_scalogram_1d, plot_scalogram_1d
from signals import make_pure_sine, make_pulse, make_exp_chirp
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
                            wavelet_bank: Union[List[T], nn.ParameterList],
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
                           avg_win: Optional[int] = None,
                           squeeze_channels: bool = True,
                           reflect_t: bool = False) -> (T, List[float], List[T]):
    assert x.ndim == 3
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr_t=sr)
    wavelet_bank, _, freqs_t, _ = make_wavelet_bank(mw, J, Q, highest_freq, reflect_t=reflect_t)
    y = _calc_scat_transform_1d(x, sr, wavelet_bank, freqs_t, should_avg, avg_win, squeeze_channels)
    return y, freqs_t, wavelet_bank


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
        wavelet_bank, _, freqs_t, _ = make_wavelet_bank(mw,
                                                        n_octaves_t=1,
                                                        steps_per_octave_t=Q,
                                                        include_lowest_octave_t=include_lowest_octave)
        octave = calc_scalogram_1d(curr_x, wavelet_bank, take_modulus=True)
        octave = average_td(octave, curr_avg_win)
        octaves.append(octave)
        freqs_all.extend(freqs_t)
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
    wavelet_bank, _, freqs_t, _ = make_wavelet_bank(mw, J, Q, highest_freq)
    y_s = []
    freqs_out = []
    # for wavelet, freq_t in tqdm(zip(wavelet_bank, freqs_t)):  # TODO(cm): tmp
    for wavelet, freq_t in zip(wavelet_bank, freqs_t):
        band_freqs = [f_x for f_x in freqs_x if f_x >= 2 * freq_t]  # TODO(cm): check what condition is correct
        n_bands = len(band_freqs)
        if n_bands == 0:
            continue
        curr_x = x[:, :n_bands, :]
        curr_wb = [wavelet]
        curr_freqs = [freq_t]
        y = _calc_scat_transform_1d(curr_x, sr, curr_wb, curr_freqs, should_avg, avg_win, squeeze_channels=False)
        if should_pad:
            pad_n = x.size(1) - n_bands
            y = F.pad(y, (0, 0, 0, pad_n))
        y_s.append(y)
        # plt.imshow(y.squeeze(0).squeeze(0).numpy(), aspect="auto")
        # plt.show()
        freqs_out.append(freq_t)

    return y_s, freqs_out


if __name__ == "__main__":
    start_n = 0
    n_samples = 2 ** 16
    # n_samples = 4096

    audio_path = "../data/flute.wav"
    flute_audio, sr = torchaudio.load(audio_path)
    flute_audio = flute_audio[:, start_n:n_samples]
    flute_audio = tr.mean(flute_audio, dim=0)

    audio = flute_audio

    sr = 48000
    audio_1 = make_pure_sine(n_samples, sr, freq=4000, amp=1.0)
    audio_2 = make_pulse(n_samples, center_loc=0.5, dur_samples=128, amp=4.0)
    audio_3 = make_exp_chirp(n_samples, sr, start_freq=20, end_freq=20000, amp=1.0)
    audio = audio_1 + audio_2 + audio_3

    audio = audio.view(1, 1, -1)

    J_1 = 12
    Q_1 = 12
    highest_freq = None
    # highest_freq = 6000

    log.info(f"in audio.shape = {audio.shape}")
    scalogram, freqs, wavelet_bank = calc_scat_transform_1d(audio,
                                                            sr,
                                                            J_1,
                                                            Q_1,
                                                            highest_freq=highest_freq)

    log.info(f"scalogram shape = {scalogram.shape}")
    # log.info(f"scalogram mean = {tr.mean(scalogram)}")
    # log.info(f"scalogram std = {tr.std(scalogram)}")
    # log.info(f"scalogram max = {tr.max(scalogram)}")
    # log.info(f"scalogram min = {tr.min(scalogram)}")
    log.info(f"scalogram energy = {MorletWavelet.calc_energy(scalogram)}")
    mean = tr.mean(scalogram)
    std = tr.std(scalogram)
    scalogram_to_plot = tr.clip(scalogram, mean - (4 * std), mean + (4 * std))
    plot_scalogram_1d(scalogram_to_plot[0], title="scalo", dt=None, freqs=freqs, n_y_ticks=J_1)

    # scalogram_fast, freqs_fast = calc_scat_transform_1d_fast(audio, sr, J_1, Q_1)
    # log.info(f"scalogram_fast shape = {scalogram_fast.shape}")
    # # log.info(f"scalogram_fast mean = {tr.mean(scalogram_fast)}")
    # # log.info(f"scalogram_fast std = {tr.std(scalogram_fast)}")
    # # log.info(f"scalogram_fast max = {tr.max(scalogram_fast)}")
    # # log.info(f"scalogram_fast min = {tr.min(scalogram_fast)}")
    # log.info(f"scalogram_fast energy = {MorletWavelet.calc_energy(scalogram_fast)}")
    # plot_scalogram_1d(scalogram_fast[0], title="scalo fast", dt=None, freqs_t=freqs_fast)
    # exit()

    J_2_t = 12
    Q_2_t = 1
    # highest_freq_t = None
    highest_freq_t = 12000

    should_avg_t = False
    log.info(f"should_avg_t = {should_avg_t}")
    avg_win_f = None
    avg_win_t = None

    y_t, freqs_t = calc_scat_transform_1d_jagged(scalogram,
                                                 sr,
                                                 freqs,
                                                 J_2_t,
                                                 Q_2_t,
                                                 should_avg=should_avg_t,
                                                 highest_freq=highest_freq_t,
                                                 avg_win=avg_win_t,
                                                 should_pad=True)
    log.info(f"freqs_t = {freqs_t}")
    for y, freq in zip(y_t, freqs_t):
        y = y.squeeze(0).squeeze(0).numpy()
        plt.imshow(y, aspect="auto", interpolation="none", cmap="OrRd")
        plt.title(f"2nd order time scattering, freq = {freq:.2f}")
        plt.show()
    exit()
