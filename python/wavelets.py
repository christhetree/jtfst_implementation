import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class DiscreteWavelet(ABC):
    def __init__(self,
                 sr_t: float = 44100,
                 sr_f: Optional[float] = None) -> None:
        self.sr_t = sr_t
        if sr_f is None:
            self.sr_f = sr_t
        else:
            self.sr_f = sr_f

        self.dt = 1.0 / self.sr_t
        self.df = 1.0 / self.sr_f

        nyquist_t = self.sr_t / 2.0
        self.min_scale_t = self.freq_to_scale(nyquist_t)
        nyquist_f = self.sr_f / 2.0
        self.min_scale_f = self.freq_to_scale(nyquist_f)

    @abstractmethod
    def period_to_scale(self, period: float) -> float:
        pass

    @abstractmethod
    def make_t_from_scale(self, s: float, dt: float) -> T:
        pass

    @abstractmethod
    def y_1d(self, t: T, s: float) -> T:
        pass

    def freq_to_scale(self, freq: float) -> float:
        return self.period_to_scale(1.0 / freq)

    def normalize_to_unit_energy(self, wavelet: T) -> T:
        energy = DiscreteWavelet.calc_energy(wavelet)
        wavelet *= energy ** -0.5
        wavelet *= wavelet.size(-1) ** -0.5
        if wavelet.ndim == 2:
            wavelet *= wavelet.size(-2) ** -0.5
        return wavelet

    def y_2d(self, t_1: T, t_2: T, s_1: float = 1.0, s_2: float = 1.0) -> T:
        assert t_1.ndim == 1
        assert t_2.ndim == 1
        assert t_1.size(0) * t_2.size(0) <= 2 ** 26  # Larger than this is too big for reasonable compute times
        y_1 = self.normalize_to_unit_energy(self.y_1d(t_1, s_1))
        y_2 = self.normalize_to_unit_energy(self.y_1d(t_2, s_2))
        y = tr.outer(y_1, y_2)
        return y

    def create_1d_wavelet_from_scale(self, s_t: float = 1.0, reflect: bool = False, normalize: bool = True) -> T:
        with tr.no_grad():
            assert s_t >= self.min_scale_t
            t = self.make_t_from_scale(s_t, self.dt)
            if reflect:
                t = -t
            wavelet = self.y_1d(t, s_t)
            if normalize:
                wavelet = self.normalize_to_unit_energy(wavelet)
            return wavelet

    def create_2d_wavelet_from_scale(self,
                                     s_f: float = 1.0,
                                     s_t: float = 1.0,
                                     reflect: bool = False,
                                     normalize: bool = True) -> T:
        with tr.no_grad():
            assert s_f >= self.min_scale_f
            assert s_t >= self.min_scale_t
            t_f = self.make_t_from_scale(s_f, self.df)
            t_t = self.make_t_from_scale(s_t, self.dt)
            if reflect:
                t_t = -t_t
            wavelet = self.y_2d(t_f, t_t, s_f, s_t)
            if normalize:
                wavelet = self.normalize_to_unit_energy(wavelet)
            return wavelet

    @staticmethod
    def calc_energy(signal: T) -> float:
        return tr.sum(tr.abs(signal) ** 2).item()


class MorletWavelet(DiscreteWavelet):
    def __init__(self,
                 sr_t: float = 44100,
                 sr_f: Optional[float] = None,
                 n_sig: float = 3.0,
                 w: Optional[float] = None):
        if w is None:
            w = MorletWavelet.freq_to_w_at_s(freq=1.0, s=1.0)
        self.w = tr.tensor(w)
        self.n_sig = n_sig  # Contains >= 99.7% of the wavelet if >= 3.0

        self.a = tr.tensor(tr.pi ** -0.25)
        self.b = tr.exp(-0.5 * (self.w ** 2))
        super().__init__(sr_t, sr_f)

    def period_to_scale(self, period: float) -> float:
        return period * (self.w + ((2.0 + (self.w ** 2)) ** 0.5)) / (4 * tr.pi)

    def make_t_from_scale(self, s: float, dt: float) -> T:
        M = int((self.n_sig * s) / dt)
        t = tr.arange(-M, M + 1) * dt
        return t

    def y_1d(self, t: T, s: float = 1.0) -> T:
        assert t.ndim == 1
        x = t / s
        y = self.a * (tr.exp(1j * self.w * x) - self.b) * tr.exp(-0.5 * (x ** 2))
        return y

    def scale_to_period(self, s: float) -> float:
        return (4 * tr.pi * s) / (self.w + ((2.0 + (self.w ** 2)) ** 0.5))

    def scale_to_freq(self, s: float) -> float:
        period = self.scale_to_period(s)
        return 1.0 / period

    @staticmethod
    def period_to_w_at_s(period: float, s: float = 1.0) -> float:
        return (((4 * tr.pi * s) ** 2) - (2 * (period ** 2))) / (8 * tr.pi * period * s)

    @staticmethod
    def freq_to_w_at_s(freq: float, s: float = 1.0) -> float:
        return MorletWavelet.period_to_w_at_s(1.0 / freq, s)
