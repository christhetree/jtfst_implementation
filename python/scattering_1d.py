import logging
import os
from typing import Optional, List, Union

import torch as tr
import torch.nn.functional as F
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T, nn
from tqdm import tqdm

from dwt import average_td, dwt_1d
from filterbanks import make_wavelet_bank
from signals import make_pure_sine, make_pulse, make_exp_chirp
from util import plot_scalogram
from wavelets import MorletWavelet, DiscreteWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ScatTransform1D(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int = 12,
                 Q: int = 12,
                 should_avg: bool = False,
                 avg_win: Optional[int] = None,
                 highest_freq: Optional[float] = None,
                 squeeze_channels: bool = True,
                 reflect_t: bool = False) -> None:
        super().__init__()
        self.sr = sr
        self.J = J
        self.Q = Q
        self.should_avg = should_avg
        self.avg_win = avg_win
        self.highest_freq = highest_freq
        self.squeeze_channels = squeeze_channels
        self.reflect_t = reflect_t

        mw = MorletWavelet(sr_t=sr)
        wavelet_bank, _, freqs_t, _ = make_wavelet_bank(mw, J, Q, highest_freq, reflect_t=reflect_t)
        self.wavelet_bank = nn.ParameterList(wavelet_bank)
        self.freqs_t = freqs_t

    def forward(self, x: T) -> (T, List[float]):
        with tr.no_grad():
            y = ScatTransform1D.calc_scat_transform_1d(x,
                                                       self.sr,
                                                       self.wavelet_bank,
                                                       self.freqs_t,
                                                       self.should_avg,
                                                       self.avg_win,
                                                       self.squeeze_channels)
            assert y.size(1) == len(self.freqs_t)
            return y, self.freqs_t

    @staticmethod
    def calc_scat_transform_1d(x: T,
                               sr: float,
                               wavelet_bank: Union[List[T], nn.ParameterList],
                               freqs: List[float],
                               should_avg: bool = False,
                               avg_win: Optional[int] = None,
                               squeeze_channels: bool = True) -> T:
        assert x.ndim == 3
        assert len(wavelet_bank) == len(freqs)
        y = dwt_1d(x, wavelet_bank, take_modulus=True, squeeze_channels=squeeze_channels)

        if not should_avg:
            return y

        if avg_win is None:
            lowest_freq = freqs[-1]
            assert sr % lowest_freq == 0
            avg_win = int(sr / lowest_freq)
            log.info(f"defaulting avg_win to {avg_win} samples ({lowest_freq:.2f} Hz at {sr:.0f} SR)")

        y = average_td(y, avg_win, dim=-1)
        return y


class ScatTransform1DJagged(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int = 12,
                 Q: int = 12,
                 should_avg: bool = False,
                 avg_win: Optional[int] = None,
                 highest_freq: Optional[float] = None,
                 reflect_t: bool = False,
                 should_pad: bool = False) -> None:
        super().__init__()
        self.sr = sr
        self.J = J
        self.Q = Q
        self.should_avg = should_avg
        if should_avg:
            assert avg_win is not None, "ScatTransform1DJagged cannot infer the averaging window automatically"
        self.avg_win = avg_win
        self.highest_freq = highest_freq
        self.reflect_t = reflect_t
        self.should_pad = should_pad

        mw = MorletWavelet(sr_t=sr)
        wavelet_bank, _, freqs_t, _ = make_wavelet_bank(mw, J, Q, highest_freq, reflect_t=reflect_t)
        self.wavelet_bank = nn.ParameterList(wavelet_bank)
        self.freqs_t = freqs_t

    def forward(self, x: T, freqs_x: List[float]) -> (List[T], List[float]):
        with tr.no_grad():
            assert x.ndim == 3
            assert x.size(1) == len(freqs_x)
            y_s = []
            freqs_t = []
            for wavelet, freq_t in tqdm(zip(self.wavelet_bank, self.freqs_t)):
                # TODO(cm): check what condition is correct
                band_freqs = [f_x for f_x in freqs_x if f_x >= 2 * freq_t]
                n_bands = len(band_freqs)
                if n_bands == 0:
                    continue
                curr_x = x[:, :n_bands, :]
                curr_wb = [wavelet]
                curr_freqs = [freq_t]
                y = ScatTransform1D.calc_scat_transform_1d(curr_x,
                                                           self.sr,
                                                           curr_wb,
                                                           curr_freqs,
                                                           self.should_avg,
                                                           self.avg_win,
                                                           squeeze_channels=False)
                if self.should_pad:
                    pad_n = x.size(1) - n_bands
                    y = F.pad(y, (0, 0, 0, pad_n))
                y = y.squeeze(1)
                y_s.append(y)
                freqs_t.append(freq_t)
            assert len(y_s) == len(freqs_t)
            return y_s, freqs_t


class ScatTransform1DSubsampling(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int = 12,
                 Q: int = 12,
                 squeeze_channels: bool = True,
                 reflect_t: bool = False) -> None:
        super().__init__()
        self.sr = sr
        self.J = J
        self.Q = Q
        self.squeeze_channels = squeeze_channels
        self.reflect_t = reflect_t

        wavelet_banks = []
        avg_wins = []
        freqs_all = []

        curr_sr = sr
        curr_avg_win = 2 ** (J + 1)
        for curr_j in range(J):
            if curr_j == J - 1:
                include_lowest_octave = True
            else:
                include_lowest_octave = False
            mw = MorletWavelet(sr_t=curr_sr)
            wavelet_bank, _, freqs_t, _ = make_wavelet_bank(mw,
                                                            n_octaves_t=1,
                                                            steps_per_octave_t=Q,
                                                            include_lowest_octave_t=include_lowest_octave,
                                                            reflect_t=reflect_t)
            wavelet_bank = nn.ParameterList(wavelet_bank)
            wavelet_banks.append(wavelet_bank)
            avg_wins.append(curr_avg_win)
            freqs_all.extend(freqs_t)
            curr_sr /= 2
            curr_avg_win //= 2

        self.wavelet_banks = nn.ParameterList(wavelet_banks)
        self.avg_wins = avg_wins
        self.freqs_t = freqs_all

    def forward(self, x: T) -> (T, List[float]):
        with tr.no_grad():
            octaves = []
            for wavelet_bank, avg_win in zip(self.wavelet_banks, self.avg_wins):
                octave = dwt_1d(x, wavelet_bank, take_modulus=True, squeeze_channels=self.squeeze_channels)
                octave = average_td(octave, avg_win)
                octaves.append(octave)
                x = average_td(x, avg_win=2, hop_size=2)
            y = tr.cat(octaves, dim=1)
            return y, self.freqs_t


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

    J_1 = 4
    Q_1 = 12
    highest_freq = None
    # highest_freq = 6000
    should_avg = True
    avg_win = 2 ** 5

    log.info(f"in audio.shape = {audio.shape}")
    scat_transform_1d = ScatTransform1D(sr,
                                        J_1,
                                        Q_1,
                                        should_avg=should_avg,
                                        avg_win=avg_win,
                                        highest_freq=highest_freq,
                                        squeeze_channels=True)
    scalogram, freqs = scat_transform_1d(audio)
    log.info(f"Lowest freq = {freqs[-1]:.2f}")

    log.info(f"scalogram shape = {scalogram.shape}")
    log.info(f"scalogram energy = {DiscreteWavelet.calc_energy(scalogram)}")
    mean = tr.mean(scalogram)
    std = tr.std(scalogram)
    scalogram_to_plot = tr.clip(scalogram.squeeze(2), mean - (4 * std), mean + (4 * std))
    plot_scalogram(scalogram_to_plot[0], title="scalo", dt=None, freqs=freqs, n_y_ticks=J_1)
    # exit()

    scat_transform_1d_subsampling = ScatTransform1DSubsampling(sr, J_1, Q_1, squeeze_channels=True)

    scalogram_fast, freqs_fast = scat_transform_1d_subsampling(audio)
    log.info(f"scalogram_fast shape = {scalogram_fast.shape}")
    log.info(f"scalogram_fast energy = {DiscreteWavelet.calc_energy(scalogram_fast)}")
    mean = tr.mean(scalogram_fast)
    std = tr.std(scalogram_fast)
    scalogram_to_plot = tr.clip(scalogram_fast.squeeze(2), mean - (4 * std), mean + (4 * std))
    plot_scalogram(scalogram_to_plot[0], title="scalo_subsampling", dt=None, freqs=freqs_fast, n_y_ticks=J_1)
    # exit()

    scat_transform_1d_jagged = ScatTransform1DJagged(sr,
                                                     J_1,
                                                     Q_1,
                                                     should_avg=should_avg,
                                                     avg_win=avg_win,
                                                     highest_freq=highest_freq,
                                                     should_pad=True)
    scalogram_jagged, freqs_jagged = scat_transform_1d_jagged(audio, [sr])
    scalogram_jagged = tr.cat(scalogram_jagged, dim=1)
    log.info(f"scalogram_jagged shape = {scalogram_jagged.shape}")
    log.info(f"scalogram_jagged energy = {DiscreteWavelet.calc_energy(scalogram_jagged)}")
    mean = tr.mean(scalogram_jagged)
    std = tr.std(scalogram_jagged)
    scalogram_to_plot = tr.clip(scalogram_jagged.squeeze(2), mean - (4 * std), mean + (4 * std))
    plot_scalogram(scalogram_to_plot[0], title="scalo_jagged", dt=None, freqs=freqs_jagged, n_y_ticks=J_1)
    # exit()

    J_2_t = 13
    Q_2_t = 1
    # highest_freq_t = None
    highest_freq_t = 12000

    should_avg_t = False
    log.info(f"should_avg_t = {should_avg_t}")
    avg_win_t = None

    scat_transform_1d_jagged = ScatTransform1DJagged(sr,
                                                     J_2_t,
                                                     Q_2_t,
                                                     should_avg=False,
                                                     avg_win=avg_win,
                                                     highest_freq=highest_freq_t,
                                                     should_pad=True)
    y_t, freqs_t = scat_transform_1d_jagged(scalogram, freqs)
    log.info(f"y_t len = {len(y_t)}")
    log.info(f"freqs_t = {freqs_t}")
    for y, freq in zip(y_t, freqs_t):
        y = y.squeeze(0).squeeze(0).numpy()
        plt.imshow(y, aspect="auto", interpolation="none", cmap="OrRd")
        plt.title(f"2nd order time scattering, freq = {freq:.2f} Hz")
        plt.show()
    exit()
