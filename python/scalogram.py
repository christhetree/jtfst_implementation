import logging
import os
from typing import Optional, List, Union, Tuple

import matplotlib.pyplot as plt
import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from tqdm import tqdm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MorletWavelet:
    def __init__(self, w: float = 6.0, sr: float = 44100, n_sig: float = 3.0, dtype: tr.dtype = tr.complex64):
        self.w = tr.tensor(w)
        self.sr = sr
        self.n_sig = n_sig  # Contains >= 99.7% of the wavelet if >= 3.0
        self.dtype = dtype

        self.a = tr.tensor(tr.pi ** -0.25)
        self.b = tr.exp(-0.5 * (self.w ** 2))
        self.dt = 1.0 / self.sr

        nyquist = self.sr / 2.0
        self.min_scale = self.period_to_scale(1.0 / nyquist)

    def y_1d(self, t: T, s: float = 1.0) -> T:
        assert t.ndim == 1
        x = t / s
        y = self.a * (tr.exp(1j * self.w * x) - self.b) * tr.exp(-0.5 * (x ** 2))
        if y.dtype != self.dtype:
            y = y.to(self.dtype)
        return y

    def create_1d_wavelet_from_scale(self, s: float = 1.0, normalize: bool = True) -> (T, T):
        with tr.no_grad():
            assert s >= self.min_scale
            M = int((self.n_sig * s) / self.dt)
            t = tr.arange(-M, M + 1) * self.dt
            wavelet = self.y_1d(t, s)
            if normalize:
                wavelet = MorletWavelet.normalize_to_unit_energy(wavelet)
            return t, wavelet

    def y_2d(self, t_1: T, t_2: T, s_1: float = 1.0, s_2: float = 1.0, reflect: bool = False) -> T:
        assert t_1.ndim == 1
        assert t_2.ndim == 1
        assert t_1.size(0) * t_2.size(0) <= 2 ** 26  # TODO(cm)
        y_1 = MorletWavelet.normalize_to_unit_energy(self.y_1d(t_1, s_1))
        if reflect:
            y_2 = MorletWavelet.normalize_to_unit_energy(self.y_1d(-t_2, s_2))
        else:
            y_2 = MorletWavelet.normalize_to_unit_energy(self.y_1d(t_2, s_2))
        y = tr.outer(y_1, y_2)
        return y

    def create_2d_wavelet_from_scale(self,
                                     s_1: float = 1.0,
                                     s_2: float = 1.0,
                                     reflect: bool = False,
                                     normalize: bool = True) -> (T, T, T):
        with tr.no_grad():
            assert s_1 >= self.min_scale
            assert s_2 >= self.min_scale
            M_1 = int((self.n_sig * s_1) / self.dt)
            t_1 = tr.arange(-M_1, M_1 + 1) * self.dt
            M_2 = int((self.n_sig * s_2) / self.dt)
            t_2 = tr.arange(-M_2, M_2 + 1) * self.dt
            wavelet = self.y_2d(t_1, t_2, s_1, s_2, reflect=reflect)
            if normalize:  # Should already be normalized
                wavelet = MorletWavelet.normalize_to_unit_energy(wavelet)
            return t_1, t_2, wavelet

    def scale_to_period(self, s: float) -> float:
        return (4 * tr.pi * s) / (self.w + ((2.0 + (self.w ** 2)) ** 0.5))

    def scale_to_freq(self, s: float) -> float:
        period = self.scale_to_period(s)
        return 1.0 / period

    def period_to_scale(self, period: float) -> float:
        return period * (self.w + ((2.0 + (self.w ** 2)) ** 0.5)) / (4 * tr.pi)

    def freq_to_scale(self, freq: float) -> float:
        return self.period_to_scale(1.0 / freq)

    @staticmethod
    def period_to_w_at_s(period: float, s: float = 1.0) -> float:
        return (((4 * tr.pi * s) ** 2) - (2 * (period ** 2))) / (8 * tr.pi * period * s)

    @staticmethod
    def freq_to_w_at_s(period: float, s: float = 1.0) -> float:
        return MorletWavelet.period_to_w_at_s(1.0 / period, s)

    @staticmethod
    def calc_energy(signal: T) -> float:
        return tr.sum(tr.abs(signal) ** 2).item()

    @staticmethod
    def normalize_to_unit_energy(wavelet: T) -> T:
        with tr.no_grad():
            energy = MorletWavelet.calc_energy(wavelet)
            return (energy ** -0.5) * wavelet


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


def calc_scales_and_freqs(mw: MorletWavelet,
                          n_octaves: int,
                          steps_per_octave: int,
                          highest_freq: Optional[float] = None) -> (List[float], List[float]):
    assert n_octaves >= 0
    assert steps_per_octave >= 1

    if highest_freq is None:
        smallest_period = 2.0 * mw.dt
    else:
        smallest_period = 1.0 / highest_freq
        assert smallest_period / mw.dt >= 2.0

    scales = []
    periods = []

    for j in range(n_octaves + 1):
        curr_period = smallest_period * (2 ** j)
        s = mw.period_to_scale(curr_period)
        scales.append(s)
        periods.append(curr_period)
        if j == n_octaves:
            break

        for q in range(1, steps_per_octave):
            exp = j + (q / steps_per_octave)
            curr_period = smallest_period * (2 ** exp)
            s = mw.period_to_scale(curr_period)
            scales.append(s)
            periods.append(curr_period)

    freqs = [1.0 / p for p in periods]
    return scales, freqs


def make_wavelet_bank(mw: MorletWavelet,
                      n_octaves_1: int,
                      steps_per_octave_1: int,
                      n_octaves_2: Optional[int] = None,
                      steps_per_octave_2: int = 1,
                      highest_freq_1: Optional[float] = None,
                      highest_freq_2: Optional[float] = None,
                      normalize: bool = True) -> (List[T], List[Union[Tuple[float, float, int], float]]):
    scales_1, freqs_1 = calc_scales_and_freqs(mw, n_octaves_1, steps_per_octave_1, highest_freq_1)
    log.info(f"freqs_1 highest = {freqs_1[0]:.0f}")
    log.info(f"freqs_1 lowest  = {freqs_1[-1]:.0f}")

    if n_octaves_2 is not None:
        scales_2, freqs_2 = calc_scales_and_freqs(mw, n_octaves_2, steps_per_octave_2, highest_freq_2)
        log.info(f"freqs_2 highest = {freqs_2[0]:.0f}")
        log.info(f"freqs_2 lowest  = {freqs_2[-1]:.0f}")
    else:
        scales_2 = None
        freqs_2 = None

    wavelet_bank = []
    freqs = []
    if scales_2:
        for s_1, freq_1 in zip(scales_1, freqs_1):
            for s_2, freq_2 in zip(scales_2, freqs_2):
                _, _, wavelet = mw.create_2d_wavelet_from_scale(s_1, s_2, reflect=False, normalize=normalize)
                wavelet_bank.append(wavelet)
                freqs.append((freq_1, freq_2, 1))
                _, _, wavelet_reflected = mw.create_2d_wavelet_from_scale(s_1, s_2, reflect=True, normalize=normalize)
                wavelet_bank.append(wavelet_reflected)
                freqs.append((freq_1, freq_2, -1))
    else:
        for s_1, freq_1 in zip(scales_1, freqs_1):
            _, wavelet = mw.create_1d_wavelet_from_scale(s_1, normalize=normalize)
            wavelet_bank.append(wavelet)
            freqs.append(freq_1)

    return wavelet_bank, freqs


def calc_scalogram_td(audio: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert audio.ndim == 3
    n_ch = audio.size(1)

    audio_complex = audio.to(mw.dtype)
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
