import logging
import os
from typing import Optional, List

import matplotlib.pyplot as plt
import torch as tr
import torch.nn.functional as F
import torchaudio
from torch import Tensor as T
from tqdm import tqdm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MorletWavelet:
    def __init__(self, w: float = 6.0, sr: float = 44100, n_sig: float = 4.0, dtype: tr.dtype = tr.complex64):
        self.w = tr.tensor(w)
        self.sr = sr
        self.n_sig = n_sig  # Contains 99.99% of the wavelet if >= 4.0
        self.dtype = dtype

        self.a = tr.tensor(tr.pi ** -0.25)
        self.b = tr.exp(-0.5 * (self.w ** 2))
        self.dt = 1.0 / self.sr

        nyquist = self.sr / 2.0
        self.min_scale = self.fourier_period_to_scale(1.0 / nyquist)

    def y(self, t: T, s: float = 1.0) -> T:
        with tr.no_grad():
            x = t / s
            y = self.a * (tr.exp(1j * self.w * x) - self.b) * tr.exp(-0.5 * (x ** 2))
            if y.dtype != self.dtype:
                y = y.to(self.dtype)
            return y

    def create_wavelet(self, s: float = 1.0, normalize: bool = True) -> (T, T):
        assert s >= self.min_scale
        M = int((self.n_sig * s) / self.dt)
        t = tr.arange(-M, M + 1) * self.dt
        wavelet = self.y(t, s)
        if normalize:
            wavelet = MorletWavelet.normalize_to_unit_energy(wavelet)
        return t, wavelet

    def create_wavelet_from_n_points(self, n_points: float, normalize: bool = True) -> (T, T):
        assert n_points >= 2.0
        period = n_points * self.dt
        s = self.fourier_period_to_scale(period)
        assert s <= 1.0
        return self.create_wavelet(s, normalize)

    def scale_to_fourier_period(self, s: float) -> float:
        return (4 * tr.pi * s) / (self.w + ((2.0 + (self.w ** 2)) ** 0.5))

    def scale_to_freq(self, s: float) -> float:
        period = self.scale_to_fourier_period(s)
        return 1.0 / period

    def fourier_period_to_scale(self, period: float) -> float:
        return period * (self.w + ((2.0 + (self.w ** 2)) ** 0.5)) / (4 * tr.pi)

    def freq_to_scale(self, freq: float) -> float:
        return self.fourier_period_to_scale(1.0 / freq)

    @staticmethod
    def fourier_period_to_w_at_s(period: float, s: float = 1.0) -> float:
        return (((4 * tr.pi * s) ** 2) - (2 * (period ** 2))) / (8 * tr.pi * period * s)

    @staticmethod
    def freq_to_w_at_s(period: float, s: float = 1.0) -> float:
        return MorletWavelet.fourier_period_to_w_at_s(1.0 / period, s)

    @staticmethod
    def calc_energy(signal: T) -> float:
        return tr.sum(tr.abs(signal) ** 2).item()

    @staticmethod
    def normalize_to_unit_energy(wavelet: T) -> T:
        energy = MorletWavelet.calc_energy(wavelet)
        return (energy ** -0.5) * wavelet


def make_wavelet_bank(mw: MorletWavelet,
                      n_octaves: int,
                      steps_per_octave: int,
                      normalize: bool = True,
                      highest_freq: Optional[float] = None) -> (List[float], List[T]):
    assert n_octaves >= 1
    assert steps_per_octave >= 0
    n_points_all = []
    wavelet_bank = []
    highest_freq_factor = 1.0
    if highest_freq is not None:
        highest_freq_factor = 1.0 / (highest_freq * mw.dt * 2.0)

    for j in range(1, n_octaves + 1):
        n_points = highest_freq_factor * (2 ** j)
        n_points_all.append(n_points)
        _, y = mw.create_wavelet_from_n_points(n_points, normalize)
        wavelet_bank.append(y)
        if j == n_octaves:
            break

        for q in range(1, steps_per_octave):
            exp = j + (q / steps_per_octave)
            n_points = highest_freq_factor * (2 ** exp)
            n_points_all.append(n_points)
            _, y = mw.create_wavelet_from_n_points(n_points, normalize)
            wavelet_bank.append(y)

    freqs = []
    for n_points in n_points_all:
        period = n_points * mw.dt
        freq = 1.0 / period
        freqs.append(round(freq, 4))

    return freqs, wavelet_bank


def calc_scalogram(audio: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert audio.ndim == 3
    n_ch = audio.size(1)
    assert n_ch == 1  # Only support mono audio for now

    audio_complex = audio.to(mw.dtype)
    convs = []
    for wavelet in tqdm(wavelet_bank):
        kernel = wavelet.view(1, 1, -1).repeat(1, n_ch, 1)
        out = F.conv1d(audio_complex, kernel, stride=(1,), padding="same")
        convs.append(out)

    scalogram = tr.cat(convs, dim=1)
    if take_modulus:
        scalogram = tr.abs(scalogram)

    return scalogram


def plot_scalogram(scalogram: T,
                   dt: Optional[float] = None,
                   freqs: Optional[List[float]] = None,
                   n_x_ticks: int = 8,
                   n_y_ticks: int = 8) -> None:
    assert scalogram.ndim == 2
    scalogram = scalogram.detach().numpy()
    plt.imshow(scalogram, aspect="auto", interpolation="none", cmap="OrRd")
    plt.title("scalogram")

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


if __name__ == "__main__":
    # audio = tr.rand((1, 1, dur))

    audio_path = "../data/flute.wav"
    audio, audio_sr = torchaudio.load(audio_path)
    # dur = int(0.5 * audio_sr)
    dur = int(2.2 * audio_sr)
    audio = tr.mean(audio, dim=0)[:dur]
    audio = audio.view(1, 1, -1)
    # exit()

    sr = audio_sr
    normalize_wavelets = True
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    log.info(f"w = {w}")
    mw = MorletWavelet(w=w, sr=sr)

    # t, y = mw.create_wavelet()
    # log.info(f"energy = {MorletWavelet.calc_energy(y)}")
    # plt.plot(t, y, label="y_norm")
    # plt.show()
    # exit()

    # J_1 = 9
    # Q_1 = 12
    # highest_freq = 14080
    J_1 = 4   # No. of octaves
    Q_1 = 16  # Steps per octave
    highest_freq = 1760
    freqs, wavelet_bank = make_wavelet_bank(mw, J_1, Q_1, normalize_wavelets, highest_freq)
    log.info(f"lowest freq = {freqs[-1]:.0f}")
    log.info(f"highest freq = {freqs[0]:.0f}")

    scalogram = calc_scalogram(audio, wavelet_bank, take_modulus=True)
    log.info(f'scalogram mean = {tr.mean(scalogram)}')
    log.info(f'scalogram std = {tr.std(scalogram)}')
    log.info(f'scalogram max = {tr.max(scalogram)}')
    log.info(f'scalogram min = {tr.min(scalogram)}')
    plot_scalogram(scalogram[0], dt=mw.dt, freqs=freqs)
