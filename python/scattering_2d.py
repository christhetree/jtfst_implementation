import logging
import os
import time
from typing import Optional, List, Tuple, Union

import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T, nn

from dwt import dwt_2d, average_td
from filterbanks import make_wavelet_bank
from scattering_1d import ScatTransform1D
from signals import make_pure_sine, make_pulse, make_exp_chirp
from util import plot_scalogram
from wavelets import MorletWavelet, DiscreteWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ScatTransform2D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_f: int,
                 J_t: int,
                 Q_f: int = 1,
                 Q_t: int = 1,
                 should_avg_f: bool = False,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 highest_freq_f: Optional[float] = None,
                 highest_freq_t: Optional[float] = None,
                 reflect_f: bool = True) -> None:
        super().__init__()
        self.sr = sr
        self.J_f = J_f
        self.J_t = J_t
        self.Q_f = Q_f
        self.Q_t = Q_t
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t
        self.highest_freq_f = highest_freq_f
        self.highest_freq_t = highest_freq_t
        self.reflect_f = reflect_f

        mw = MorletWavelet(sr_t=sr)
        wavelet_bank, freqs_f, freqs_t, orientations = make_wavelet_bank(mw,
                                                                         J_t,
                                                                         Q_t,
                                                                         highest_freq_t,
                                                                         n_octaves_f=J_f,
                                                                         steps_per_octave_f=Q_f,
                                                                         highest_freq_f=highest_freq_f,
                                                                         reflect_f=reflect_f)
        self.wavelet_bank = nn.ParameterList(wavelet_bank)
        self.freqs = list(zip(freqs_f, freqs_t, orientations))

    def forward(self, x: T) -> (T, List[Tuple[float, float, int]]):
        with tr.no_grad():
            y = ScatTransform2D.calc_scat_transform_2d(x,
                                                       self.sr,
                                                       self.wavelet_bank,
                                                       self.freqs,
                                                       self.should_avg_f,
                                                       self.should_avg_t,
                                                       self.avg_win_f,
                                                       self.avg_win_t)
            assert y.size(1) == len(self.freqs)
            return y, self.freqs

    @staticmethod
    def calc_scat_transform_2d(x: T,
                               sr: float,
                               wavelet_bank: Union[List[T], nn.ParameterList],
                               freqs: List[Tuple[float, float, int]],
                               should_avg_f: bool = False,
                               should_avg_t: bool = True,
                               avg_win_f: Optional[int] = None,
                               avg_win_t: Optional[int] = None) -> T:
        assert x.ndim == 3
        assert len(wavelet_bank) == len(freqs)
        y = dwt_2d(x, wavelet_bank, take_modulus=True)

        if not should_avg_f and not should_avg_t:
            return y

        lowest_freq_f, lowest_freq_t, _ = freqs[-1]
        if should_avg_t:
            if avg_win_t is None:
                assert sr % lowest_freq_t == 0
                avg_win_t = int(sr / lowest_freq_t)
                log.info(f"defaulting avg_win_t to {avg_win_t} samples ({lowest_freq_t:.2f} Hz at {sr:.0f} SR)")

            max_wavelet_len_t = max([w.size(1) for w in wavelet_bank])
            if avg_win_t > (max_wavelet_len_t + 1) // 6:
                log.warning(
                    "Time averaging window is suspiciously large (probably greater than the lowest central freq)")
            y = average_td(y, avg_win_t, dim=-1)

        if should_avg_f:
            if avg_win_f is None:
                log.info(f"should_avg_f is True, but avg_win_f is None, using a heuristic value of 2")
                avg_win_f = 2

            log.info(f"avg_win_f = {avg_win_f}")
            max_wavelet_len_f = max([w.size(0) for w in wavelet_bank])
            if avg_win_f > (max_wavelet_len_f + 1) // 6:
                log.warning(
                    "Freq averaging window is suspiciously large (probably greater than the lowest central freq)")
            y = average_td(y, avg_win_f, dim=-2)

        return y


# TODO(cm): implement ability to set highest frequency
class ScatTransform2DSubsampling(nn.Module):
    def __init__(self,
                 sr: float,
                 J_f: int,
                 J_t: int,
                 Q_f: int = 1,
                 Q_t: int = 1,
                 should_avg_f: bool = False,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 reflect_f: bool = True) -> None:
        super().__init__()
        self.sr = sr
        self.J_f = J_f
        self.J_t = J_t
        self.Q_f = Q_f
        self.Q_t = Q_t
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_t = avg_win_t
        self.reflect_f = reflect_f

        if should_avg_f:
            if avg_win_f is None:
                log.info(f"should_avg_f is True, but avg_win_f is None, using a heuristic value of 2")
                avg_win_f = 2
        self.avg_win_f = avg_win_f

        curr_sr_t = sr
        curr_highest_freq_t = sr / 2

        if should_avg_t:
            if avg_win_t is None:
                curr_avg_win_t = 2 ** J_t  # TODO(cm): check
                log.info(f"defaulting avg_win_t to {curr_avg_win_t} samples "
                         f"({sr / curr_avg_win_t:.2f} Hz at {sr:.0f} SR)")
            else:
                curr_avg_win_t = avg_win_t
        else:
            curr_avg_win_t = 1

        wavelet_banks = []
        avg_wins_t = []
        freqs_all = []
        for curr_j_t in range(J_t):
            if curr_j_t == J_t - 1:
                include_lowest_octave_t = True
            else:
                include_lowest_octave_t = False

            mw = MorletWavelet(sr_t=curr_sr_t, sr_f=sr)
            wavelet_bank, freqs_f, freqs_t, orientations = make_wavelet_bank(
                mw,
                n_octaves_t=1,
                steps_per_octave_t=Q_t,
                highest_freq_t=curr_highest_freq_t,
                include_lowest_octave_t=include_lowest_octave_t,
                n_octaves_f=J_f,
                steps_per_octave_f=Q_f,
                reflect_f=reflect_f
            )
            wavelet_bank = nn.ParameterList(wavelet_bank)
            wavelet_banks.append(wavelet_bank)
            avg_wins_t.append(curr_avg_win_t)
            freqs = list(zip(freqs_f, freqs_t, orientations))
            freqs_all.extend(freqs)
            if curr_avg_win_t > 1:
                curr_sr_t /= 2
                assert curr_avg_win_t % 2 == 0
                curr_avg_win_t //= 2
            curr_highest_freq_t /= 2

        self.wavelet_banks = nn.ParameterList(wavelet_banks)
        self.avg_wins_t = avg_wins_t
        self.freqs = freqs_all

    def forward(self, x: T) -> (T, List[Tuple[float, float, int]]):
        with tr.no_grad():
            octaves = []
            for wavelet_bank, avg_win_t in zip(self.wavelet_banks, self.avg_wins_t):
                octave = dwt_2d(x, wavelet_bank, take_modulus=True)
                octave = average_td(octave, avg_win_t, dim=-1)
                if self.should_avg_f:
                    octave = average_td(octave, self.avg_win_f, dim=-2)
                octaves.append(octave)
                if avg_win_t > 1:
                    x = average_td(x, avg_win=2, dim=-1)
            y = tr.cat(octaves, dim=1)
            assert y.size(1) == len(self.freqs)
            return y, self.freqs


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

    scat_transform_1d = ScatTransform1D(sr,
                                        J_1,
                                        Q_1,
                                        should_avg=False,
                                        highest_freq=highest_freq,
                                        squeeze_channels=True)
    log.info(f"in audio.shape = {audio.shape}")
    scalogram, freqs_t = scat_transform_1d(audio)
    log.info(f"scalogram shape = {scalogram.shape}")
    mean = tr.mean(scalogram)
    std = tr.std(scalogram)
    scalogram_to_plot = tr.clip(scalogram, mean - (4 * std), mean + (4 * std))
    plot_scalogram(scalogram_to_plot[0], title="scalo", dt=None, freqs=freqs_t, n_y_ticks=J_1)
    # exit()

    J_2_f = 4
    Q_2_f = 1
    highest_freq_f = None
    J_2_t = 12
    Q_2_t = 1
    highest_freq_t = None

    should_avg_f = False
    should_avg_t = True
    log.info(f"should_avg_f = {should_avg_f}")
    log.info(f"should_avg_t = {should_avg_t}")
    avg_win_f = None
    # avg_win_f = 2 ** 2
    # avg_win_t = None
    avg_win_t = 2 ** 11

    pic_idx = -6

    start_t = time.perf_counter()
    # scat_transform_2d = ScatTransform2D(sr,
    #                                     J_2_f,
    #                                     J_2_t,
    #                                     Q_2_f,
    #                                     Q_2_t,
    #                                     should_avg_f=should_avg_f,
    #                                     should_avg_t=should_avg_t,
    #                                     avg_win_f=avg_win_f,
    #                                     avg_win_t=avg_win_t,
    #                                     highest_freq_f=highest_freq_f,
    #                                     highest_freq_t=highest_freq_t)
    # jtfst_fast, freqs_2_fast = scat_transform_2d(scalogram)
    scat_transform_2d = ScatTransform2DSubsampling(sr,
                                                   J_2_f,
                                                   J_2_t,
                                                   Q_2_f,
                                                   Q_2_t,
                                                   should_avg_f=should_avg_f,
                                                   should_avg_t=should_avg_t,
                                                   avg_win_f=avg_win_f,
                                                   avg_win_t=avg_win_t)
    jtfst_fast, freqs_2_fast = scat_transform_2d(scalogram)

    end_t = time.perf_counter()
    log.info(f"elapsed time = {end_t - start_t:.2f} seconds")
    log.info(f"lowest_freq_t = {freqs_2_fast[-1][1]:.2f}")
    log.info(f"jtfst_fast shape = {jtfst_fast.shape}")
    log.info(f"jtfst_fast energy = {DiscreteWavelet.calc_energy(jtfst_fast)}")
    mean = tr.mean(jtfst_fast)
    std = tr.std(jtfst_fast)
    jtfst_fast = tr.clip(jtfst_fast, mean - (4 * std), mean + (4 * std))
    pic = jtfst_fast[0, pic_idx, :, :].detach().numpy()
    plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    plt.title("jtfst_fast")
    plt.show()
    exit()

    n_rows = len(freqs_2_fast) // 2
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx, (freq_f, freq_t, theta) in enumerate(freqs_2_fast):
        curr_ax = ax[idx // 2, idx % 2]
        pic = jtfst_fast[0, idx, :, :].squeeze().detach().numpy()
        curr_ax.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
        curr_ax.set_title(f"freq_f = {freq_f:.0f}, freq_t = {freq_t:.0f}, theta = {theta}")
    plt.show()
