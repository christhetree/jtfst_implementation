import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MorletWavelet:
    def __init__(self, w: float = 6.0, sr: float = 44100, n_sig: float = 4.0):
        self.w = tr.tensor(w)
        self.sr = sr
        self.n_sig = n_sig  # Contains 99.99% of the wavelet if >= 4.0

        self.a = tr.tensor(tr.pi ** -0.25)
        self.b = tr.exp(-0.5 * (self.w ** 2))
        self.dt = 1.0 / self.sr

        nyquist = self.sr / 2.0
        self.min_scale = self.fourier_period_to_scale(1.0 / nyquist)

    def y(self, t: T, s: float = 1.0) -> T:
        with tr.no_grad():
            x = t / s
            return self.a * (tr.exp(1j * self.w * x) - self.b) * tr.exp(-0.5 * (x ** 2))

    def create_wavelet(self, s: float = 1.0) -> (T, T):
        assert s >= self.min_scale
        M = int((self.n_sig * s) / self.dt)
        t = tr.arange(-M, M + 1) * self.dt
        return t, self.y(t, s)

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
    def normalize_to_unit_energy(wavelet: T) -> T:
        energy = tr.sum(tr.abs(wavelet) ** 2)
        return (energy ** -0.5) * wavelet


def calc_scalogram(x: T, wavelet: Optional[T] = None) -> T:
    pass


if __name__ == "__main__":
    sr = 44100
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    print(f"w = {w}")
    mw = MorletWavelet(w=w, sr=sr)
    s = mw.freq_to_scale(22050.0)
    t, y = mw.create_wavelet(s)
    y_norm = mw.normalize_to_unit_energy(y)
    y = y.real.numpy()
    y_norm = y_norm.real.numpy()
    plt.plot(t, y_norm, label="y_norm")
    plt.legend()
    plt.show()
    exit()

    log.info("Calculating scalogram")
    batch_size = 1
    n_ch = 2
    n_samples = 48000
    audio = tr.rand((batch_size, n_ch, n_samples))
    scalogram = calc_scalogram(audio)
