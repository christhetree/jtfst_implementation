import logging
import os
from typing import Optional, List, Tuple

import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T, nn

from dwt import average_td
from scattering_1d import ScatTransform1D, ScatTransform1DJagged
from scattering_2d import ScatTransform2DSubsampling
from signals import make_pure_sine, make_pulse, make_exp_chirp
from util import plot_scalogram
from wavelets import DiscreteWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# TODO(cm): figure out why reflect_f has no effect
class JTFST1D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_1: int = 12,
                 J_2_f: int = 4,
                 J_2_t: int = 12,
                 Q_1: int = 16,
                 Q_2_f: int = 1,
                 Q_2_t: int = 1,
                 should_avg_f: bool = True,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 reflect_f: bool = False) -> None:
        super().__init__()
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t
        self.audio_to_scalogram = ScatTransform1D(sr, J_1, Q_1, should_avg=False, squeeze_channels=True)
        self.conv_t = ScatTransform1DJagged(sr, J_2_t, Q_2_t, should_avg=False, should_pad=True)
        self.conv_f = ScatTransform1D(sr,
                                      J_2_f,
                                      Q_2_f,
                                      should_avg=should_avg_f,
                                      avg_win=avg_win_f,
                                      squeeze_channels=False,
                                      reflect_t=reflect_f)

    def forward(self, x: T) -> (T, List[float], T, List[Tuple[float, float, int]]):
        scalogram, freqs_1, _ = self.audio_to_scalogram(x)
        y_t, freqs_t, _ = self.conv_t(scalogram, freqs_1)

        jtfst_s = []
        freqs_2 = []
        for y, freq_t in zip(y_t, freqs_t):
            y = tr.swapaxes(y, 1, 2)
            jtfst, freqs_f, orientations = self.conv_f(y)
            jtfst = tr.swapaxes(jtfst, 2, 3)
            jtfst_s.append(jtfst)
            for freq_f, orientation in zip(freqs_f, orientations):
                freqs_2.append((freq_f, freq_t, orientation))

        jtfst = tr.cat(jtfst_s, dim=1)
        if self.should_avg_t:
            avg_win_t = self.avg_win_t
            if avg_win_t is None:
                lowest_freq_t = freqs_t[-1]
                assert self.sr % lowest_freq_t == 0
                avg_win_t = int(self.sr / lowest_freq_t)
                log.info(f"defaulting avg_win_t to {avg_win_t} samples ({lowest_freq_t:.2f} Hz at {self.sr:.0f} SR)")

            jtfst = average_td(jtfst, avg_win_t, dim=-1)

        return scalogram, freqs_1, jtfst, freqs_2


class JTFST2D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_1: int = 12,
                 J_2_f: int = 4,
                 J_2_t: int = 12,
                 Q_1: int = 16,
                 Q_2_f: int = 1,
                 Q_2_t: int = 1,
                 should_avg_f: bool = True,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 reflect_f: bool = True) -> None:
        super().__init__()
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t
        self.audio_to_scalogram = ScatTransform1D(sr, J_1, Q_1, should_avg=False, squeeze_channels=True)
        self.scalogram_to_jtfst = ScatTransform2DSubsampling(sr,
                                                             J_2_f,
                                                             J_2_t,
                                                             Q_2_f,
                                                             Q_2_t,
                                                             should_avg_f,
                                                             should_avg_t,
                                                             avg_win_f,
                                                             avg_win_t,
                                                             reflect_f)

    def forward(self, x: T) -> (T, List[float], T, List[Tuple[float, float, int]]):
        scalogram, freqs_1, _ = self.audio_to_scalogram(x)
        jtfst, freqs_2 = self.scalogram_to_jtfst(scalogram)
        return scalogram, freqs_1, jtfst, freqs_2


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
    J_2_f = 2
    J_2_t = 12
    Q_1 = 16
    Q_2_f = 1
    Q_2_t = 1
    should_avg_f = False
    should_avg_t = True
    avg_win_f = 4  # Average across 25% of an octave if Q_1 == 16
    avg_win_t = 2 ** 9
    reflect_f = False

    jtfst_class = JTFST1D
    # jtfst_class = JTFST2D
    jtfst_func = jtfst_class(sr,
                             J_1=J_1,
                             J_2_f=J_2_f,
                             J_2_t=J_2_t,
                             Q_1=Q_1,
                             Q_2_f=Q_2_f,
                             Q_2_t=Q_2_t,
                             should_avg_f=should_avg_f,
                             should_avg_t=should_avg_t,
                             avg_win_f=avg_win_f,
                             avg_win_t=avg_win_t,
                             reflect_f=reflect_f)

    scalogram, freqs_1, jtfst, freqs_2 = jtfst_func(audio)
    plot_scalogram(scalogram[0], title="scalo", dt=None, freqs=freqs_1, n_y_ticks=12)

    pic_idx = -4
    # jtfst = jtfst[:, 3:, :, :]
    # freqs_2 = freqs_2[3:]
    log.info(f"jtfst shape = {jtfst.shape}")
    log.info(f"jtfst energy = {DiscreteWavelet.calc_energy(jtfst)}")
    mean = tr.mean(jtfst)
    std = tr.std(jtfst)
    jtfst = tr.clip(jtfst, mean - (4 * std), mean + (4 * std))
    pic = jtfst[0, pic_idx, :, :].detach().numpy()
    plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    plt.title("jtfst")
    plt.show()
    # exit()

    # Plotting
    n_rows = int(len(freqs_2) / 2 + 0.5)
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx, (freq_f, freq_t, theta) in enumerate(freqs_2):
        curr_ax = ax[idx // 2, idx % 2]
        pic = jtfst[0, idx, :, :].squeeze().detach().numpy()
        curr_ax.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
        curr_ax.set_title(f"freq_f = {freq_f:.0f}, freq_t = {freq_t:.0f}, theta = {theta}")
    plt.show()
