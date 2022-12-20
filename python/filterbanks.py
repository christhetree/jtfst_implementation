import logging
import os
from typing import Optional, List

from torch import Tensor as T

from wavelets import MorletWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_scales_and_freqs(n_octaves: int,
                          steps_per_octave: int,
                          sr: float,
                          mw: MorletWavelet,
                          highest_freq: Optional[float] = None,
                          include_lowest_octave: bool = True) -> (List[float], List[float]):
    assert n_octaves >= 1
    assert steps_per_octave >= 1

    if highest_freq is None:
        smallest_period = 2.0 / sr
    else:
        smallest_period = 1.0 / highest_freq
        assert smallest_period * sr >= 2.0

    scales = []
    periods = []

    for j in range(n_octaves):
        for q in range(steps_per_octave):
            exp = j + (q / steps_per_octave)
            curr_period = smallest_period * (2 ** exp)
            s = mw.period_to_scale(curr_period)
            scales.append(s)
            periods.append(curr_period)

        if include_lowest_octave and j == n_octaves - 1:
            curr_period = smallest_period * (2 ** (j + 1))
            s = mw.period_to_scale(curr_period)
            scales.append(s)
            periods.append(curr_period)

    freqs = [1.0 / p for p in periods]
    return scales, freqs


def make_wavelet_bank(mw: MorletWavelet,
                      n_octaves_t: int,
                      steps_per_octave_t: int = 1,
                      highest_freq_t: Optional[float] = None,
                      include_lowest_octave_t: bool = True,
                      reflect_t: bool = False,
                      n_octaves_f: Optional[int] = None,
                      steps_per_octave_f: int = 1,
                      highest_freq_f: Optional[float] = None,
                      include_lowest_octave_f: bool = True,
                      reflect_f: bool = True,
                      normalize: bool = True) -> (List[T], List[float], List[float], List[int]):
    if n_octaves_f is not None:
        scales_f, freqs_f = calc_scales_and_freqs(
            n_octaves_f, steps_per_octave_f, mw.sr_f, mw, highest_freq_f, include_lowest_octave_f)
        log.debug(f"freqs_f highest = {freqs_f[0]:.2f}")
        log.debug(f"freqs_f lowest  = {freqs_f[-1]:.2f}")
    else:
        scales_f = None
        freqs_f = None

    scales_t, freqs_t = calc_scales_and_freqs(
        n_octaves_t, steps_per_octave_t, mw.sr_t, mw, highest_freq_t, include_lowest_octave_t)
    log.debug(f"freqs_t highest = {freqs_t[0]:.2f}")
    log.debug(f"freqs_t lowest  = {freqs_t[-1]:.2f}")

    wavelet_bank = []
    freqs_f_out = []
    freqs_t_out = []
    orientations = []
    if scales_f:
        for s_t, freq_t in zip(scales_t, freqs_t):
            for s_f, freq_f in zip(scales_f, freqs_f):
                wavelet = mw.create_2d_wavelet_from_scale(s_f, s_t, reflect=False, normalize=normalize)
                wavelet_bank.append(wavelet)
                freqs_f_out.append(freq_f)
                freqs_t_out.append(freq_t)
                orientations.append(1)
                if reflect_f:
                    wavelet_reflected = mw.create_2d_wavelet_from_scale(s_f,
                                                                        s_t,
                                                                        reflect=True,
                                                                        normalize=normalize)
                    wavelet_bank.append(wavelet_reflected)
                    freqs_f_out.append(freq_f)
                    freqs_t_out.append(freq_t)
                    orientations.append(-1)
    else:
        for s_t, freq_t in zip(scales_t, freqs_t):
            wavelet = mw.create_1d_wavelet_from_scale(s_t, reflect=False, normalize=normalize)
            wavelet_bank.append(wavelet)
            freqs_t_out.append(freq_t)
            orientations.append(1)
            if reflect_t:
                wavelet_reflected = mw.create_1d_wavelet_from_scale(s_t, reflect=True, normalize=normalize)
                wavelet_bank.append(wavelet_reflected)
                freqs_t_out.append(freq_t)
                orientations.append(-1)

    return wavelet_bank, freqs_f_out, freqs_t_out, orientations
