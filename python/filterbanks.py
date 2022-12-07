import logging
import os
from typing import Optional, List, Union, Tuple

from torch import Tensor as T

from wavelets import MorletWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


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
