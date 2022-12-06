import logging
import os
from typing import Optional, List, Union, Tuple

import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MorletWavelet:
    def __init__(self,
                 w: float = 6.0,
                 sr_t: float = 44100,
                 sr_f: Optional[float] = None,
                 n_sig: float = 3.0,
                 dtype: tr.dtype = tr.complex64):
        self.w = tr.tensor(w)
        self.sr_t = sr_t
        if sr_f is None:
            self.sr_f = sr_t
        else:
            self.sr_f = sr_f
        self.n_sig = n_sig  # Contains >= 99.7% of the wavelet if >= 3.0
        self.dtype = dtype

        self.a = tr.tensor(tr.pi ** -0.25)
        self.b = tr.exp(-0.5 * (self.w ** 2))
        self.dt = 1.0 / self.sr_t
        self.df = 1.0 / self.sr_f

        nyquist_t = self.sr_t / 2.0
        self.min_scale_t = self.period_to_scale(1.0 / nyquist_t)
        nyquist_f = self.sr_f / 2.0
        self.min_scale_f = self.period_to_scale(1.0 / nyquist_f)

    def y_1d(self, t: T, s: float = 1.0) -> T:
        assert t.ndim == 1
        x = t / s
        y = self.a * (tr.exp(1j * self.w * x) - self.b) * tr.exp(-0.5 * (x ** 2))
        if y.dtype != self.dtype:
            y = y.to(self.dtype)
        return y

    def create_1d_wavelet_from_scale(self, s_t: float = 1.0, normalize: bool = True) -> (T, T):
        with tr.no_grad():
            assert s_t >= self.min_scale_t
            M = int((self.n_sig * s_t) / self.dt)
            t = tr.arange(-M, M + 1) * self.dt
            wavelet = self.y_1d(t, s_t)
            if normalize:
                wavelet = MorletWavelet.normalize_to_unit_energy(wavelet)
            return t, wavelet

    def y_2d(self, t_1: T, t_2: T, s_1: float = 1.0, s_2: float = 1.0, reflect: bool = False) -> T:
        assert t_1.ndim == 1
        assert t_2.ndim == 1
        assert t_1.size(0) * t_2.size(0) <= 2 ** 26  # TODO(cm)
        y_1 = MorletWavelet.normalize_to_unit_energy(self.y_1d(t_1, s_1))
        # TODO(cm): why is the reflection opposite?
        if reflect:
            y_2 = MorletWavelet.normalize_to_unit_energy(self.y_1d(t_2, s_2))
        else:
            y_2 = MorletWavelet.normalize_to_unit_energy(self.y_1d(-t_2, s_2))
        y = tr.outer(y_1, y_2)
        return y

    def create_2d_wavelet_from_scale(self,
                                     s_f: float = 1.0,
                                     s_t: float = 1.0,
                                     reflect: bool = False,
                                     normalize: bool = True) -> (T, T, T):
        with tr.no_grad():
            assert s_f >= self.min_scale_f
            assert s_t >= self.min_scale_t
            M_f = int((self.n_sig * s_f) / self.df)
            t_f = tr.arange(-M_f, M_f + 1) * self.df
            M_t = int((self.n_sig * s_t) / self.dt)
            t_t = tr.arange(-M_t, M_t + 1) * self.dt
            wavelet = self.y_2d(t_f, t_t, s_f, s_t, reflect=reflect)
            if normalize:  # Should already be normalized
                wavelet = MorletWavelet.normalize_to_unit_energy(wavelet)
            return t_f, t_t, wavelet

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


def calc_scales_and_freqs(n_octaves: int,
                          steps_per_octave: int,
                          sr: float,
                          mw: MorletWavelet,
                          highest_freq: Optional[float] = None) -> (List[float], List[float]):
    assert n_octaves >= 0
    assert steps_per_octave >= 1

    if highest_freq is None:
        smallest_period = 2.0 / sr
    else:
        smallest_period = 1.0 / highest_freq
        assert smallest_period * sr >= 2.0

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
                      n_octaves_t: int,
                      steps_per_octave_t: int = 1,
                      highest_freq_t: Optional[float] = None,
                      n_octaves_f: Optional[int] = None,
                      steps_per_octave_f: int = 1,
                      highest_freq_f: Optional[float] = None,
                      normalize: bool = True) -> (List[T], List[Union[Tuple[float, float, int], float]]):
    if n_octaves_f is not None:
        scales_f, freqs_f = calc_scales_and_freqs(n_octaves_f, steps_per_octave_f, mw.sr_f, mw, highest_freq_f)
        log.info(f"freqs_f highest = {freqs_f[0]:.0f}")
        log.info(f"freqs_f lowest  = {freqs_f[-1]:.0f}")
    else:
        scales_f = None
        freqs_f = None

    scales_t, freqs_t = calc_scales_and_freqs(n_octaves_t, steps_per_octave_t, mw.sr_t, mw, highest_freq_t)
    log.info(f"freqs_t highest = {freqs_t[0]:.0f}")
    log.info(f"freqs_t lowest  = {freqs_t[-1]:.0f}")

    wavelet_bank = []
    freqs = []
    if scales_f:
        for s_t, freq_t in zip(scales_t, freqs_t):
            for s_f, freq_f in zip(scales_f, freqs_f):
                _, _, wavelet = mw.create_2d_wavelet_from_scale(s_f, s_t, reflect=False, normalize=normalize)
                wavelet_bank.append(wavelet)
                freqs.append((freq_f, freq_t, 1))
                _, _, wavelet_reflected = mw.create_2d_wavelet_from_scale(s_f, s_t, reflect=True, normalize=normalize)
                wavelet_bank.append(wavelet_reflected)
                freqs.append((freq_f, freq_t, -1))
    else:
        for s_t, freq_t in zip(scales_t, freqs_t):
            _, wavelet = mw.create_1d_wavelet_from_scale(s_t, normalize=normalize)
            wavelet_bank.append(wavelet)
            freqs.append(freq_t)

    return wavelet_bank, freqs