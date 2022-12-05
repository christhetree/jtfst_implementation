import logging
import os
from typing import Optional, List

import torch as tr
import torch.nn.functional as F
import torchaudio
from torch import Tensor as T

from scalogram import MorletWavelet, make_wavelet_bank, plot_scalogram

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def scatter(x: T, scattering_window: int, dim: int = -1, hop_size: Optional[int] = None) -> T:
    # TODO(cm): do in the freq domain and add padding for last frame
    assert x.ndim >= 2
    assert scattering_window > 1
    assert x.size(dim) >= scattering_window

    if hop_size is None:
        hop_size = scattering_window
    else:
        assert hop_size <= scattering_window

    unfolded = x.unfold(dimension=dim, size=scattering_window, step=hop_size)
    out = tr.mean(unfolded, dim=-1, keepdim=False)
    return out


def scalogram_1d(signal: T,
                 wavelet_bank: List[T],
                 scattering_window: Optional[int] = None,
                 hop_size: Optional[int] = None) -> T:
    assert signal.ndim == 3
    n_ch = signal.size(1)

    max_wavelet_len = max([len(w) for w in wavelet_bank])
    max_padding = max_wavelet_len // 2
    # TODO(cm): check why we can get away with only padding the front
    signal = F.pad(signal, (max_padding, 0))
    signal_fd = tr.fft.fft(signal).unsqueeze(1)

    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 1
        left_padding = max_padding - wavelet.size(-1) // 2
        right_padding = signal_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, -1).expand(-1, n_ch, -1)
        kernel = F.pad(kernel, (left_padding, right_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=0).unsqueeze(0)
    kernels_fd = tr.fft.fft(kernels)
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    out_fd = kernels_fd * signal_fd
    out = tr.fft.ifft(out_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    out = out[:, :, :, :-max_padding]
    out = tr.sum(out, dim=2, keepdim=False)
    out = tr.abs(out)

    if scattering_window is not None:
        if scattering_window > max_wavelet_len // 6:
            log.warning("Scattering window is suspiciously large (probably greater than the lowest central freq)")
        out = scatter(out, scattering_window, hop_size=hop_size)

    return out


if __name__ == "__main__":
    # should_scatter = True
    should_scatter = False

    start_n = 48000
    n_samples = int(1.5 * 48000)

    audio_path = "../data/sine_sweep.wav"
    chirp_audio, sr = torchaudio.load(audio_path)
    chirp_audio = chirp_audio[:, start_n:start_n + n_samples]
    chirp_audio = chirp_audio.view(1, 1, -1)

    audio = chirp_audio
    # audio = tr.sin(2 * tr.pi * 440.0 * (1 / sr) * tr.arange(n_samples)).view(1, 1, -1)

    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    mw = MorletWavelet(w=w, sr=sr)

    J_1 = 5
    Q_1 = 12
    # highest_freq = None
    highest_freq = 750
    wavelet_bank, freqs = make_wavelet_bank(mw, J_1, Q_1, highest_freq_1=highest_freq)

    lowest_freq = freqs[-1]
    scattering_window = None
    if should_scatter:
        assert sr % lowest_freq == 0
        scattering_window = int(sr / lowest_freq)

    log.info(f"in audio.shape = {audio.shape}")
    log.info(f"scattering_window = {T}")
    scalogram = scalogram_1d(audio, wavelet_bank, scattering_window=scattering_window)
    log.info(f"scalogram shape = {scalogram.shape}")
    log.info(f"scalogram mean = {tr.mean(scalogram)}")
    log.info(f"scalogram std = {tr.std(scalogram)}")
    log.info(f"scalogram max = {tr.max(scalogram)}")
    log.info(f"scalogram min = {tr.min(scalogram)}")
    plot_scalogram(scalogram[0], title="chirp", dt=mw.dt, freqs=freqs)
