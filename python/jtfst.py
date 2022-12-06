import logging
import os
from typing import Optional, List

import torch as tr
import torch.nn.functional as F
import torchaudio
from torch import Tensor as T
from tqdm import tqdm

from scalogram import plot_scalogram, calc_scalogram_fd, calc_scalogram_td
from wavelets import MorletWavelet, make_wavelet_bank

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_jtfst_td(scalogram: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert scalogram.ndim == 3
    assert wavelet_bank

    scalogram = scalogram.unsqueeze(1)  # Image with 1 channel
    scalogram_complex = scalogram.to(mw.dtype)
    convs = []
    for wavelet in tqdm(wavelet_bank):
        assert wavelet.ndim == 2
        kernel = wavelet.view(1, 1, *wavelet.shape)
        out = F.conv2d(scalogram_complex, kernel, stride=(1, 1), padding="same")
        convs.append(out)

    jtfst = tr.cat(convs, dim=1)
    if take_modulus:
        jtfst = tr.abs(jtfst)

    return jtfst


def calc_jtfst_fd(scalogram: T,
                  wavelet_bank: List[T],
                  max_f_dim: Optional[int] = None,
                  max_t_dim: Optional[int] = None,
                  take_modulus: bool = True) -> T:
    assert scalogram.ndim == 3
    if max_f_dim is None:
        max_f_dim = max([w.size(0) for w in wavelet_bank])
    if max_t_dim is None:
        max_t_dim = max([w.size(1) for w in wavelet_bank])
    max_f_padding = max_f_dim // 2
    max_t_padding = max_t_dim // 2
    # TODO(cm): check why we can get away with only padding the front
    scalogram = F.pad(scalogram, (max_t_padding, 0, max_f_padding, 0))
    log.debug("scalogram fft")
    scalogram_fd = tr.fft.fft2(scalogram).unsqueeze(1)

    log.debug("making kernels")
    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 2
        top_padding = max_f_padding - wavelet.size(-2) // 2
        bottom_padding = scalogram_fd.size(-2) - wavelet.size(-2) - top_padding
        left_padding = max_t_padding - wavelet.size(-1) // 2
        right_padding = scalogram_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, *wavelet.shape)
        kernel = F.pad(kernel, (left_padding, right_padding, top_padding, bottom_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=1)
    log.debug("kernel fft")
    kernels_fd = tr.fft.fft2(kernels)
    log.debug("matmult")
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    out_fd = kernels_fd * scalogram_fd
    log.debug("jtfst ifft")
    jtfst = tr.fft.ifft2(out_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    jtfst = jtfst[:, :, :-max_f_padding, :-max_t_padding]

    if take_modulus:
        jtfst = tr.abs(jtfst)

    return jtfst


if __name__ == "__main__":
    start_n = 48000
    n_samples = int(1.5 * 48000)

    audio_path = "../data/flute.wav"
    flute_audio, audio_sr_1 = torchaudio.load(audio_path)
    flute_audio = flute_audio[:, :n_samples]
    flute_audio = tr.mean(flute_audio, dim=0)
    flute_audio = flute_audio.view(1, 1, -1)

    # # dur = int(0.2 * audio_sr)
    # # audio = tr.mean(audio, dim=0)[20000:20000 + dur]
    # dur = int(2.2 * audio_sr)
    # audio = tr.mean(audio, dim=0)[:dur]

    audio_path = "../data/sine_sweep.wav"
    chirp_audio, audio_sr_2 = torchaudio.load(audio_path)
    chirp_audio = chirp_audio[:, start_n:start_n + n_samples]
    # chirp_audio = chirp_audio[:, -n_samples:-8000]

    # chirp_audio = chirp_audio[:, -n_samples:]
    # chirp_audio = chirp_audio[:, :n_samples // 2]
    # chirp_audio = F.pad(chirp_audio, (n_samples // 2, n_samples // 2))

    chirp_audio = chirp_audio.view(1, 1, -1)
    assert audio_sr_1 == audio_sr_2

    # audio = flute_audio
    audio = chirp_audio
    # audio = tr.cat([chirp_audio, flute_audio], dim=0)
    # audio = tr.cat([tr.rand_like(flute_audio), flute_audio], dim=1)
    # audio = tr.sin(2 * tr.pi * 440.0 * (1 / audio_sr_1) * tr.arange(n_samples)).view(1, 1, -1)

    sr = audio_sr_1
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    log.info(f"w = {w}")
    mw = MorletWavelet(w=w, sr_t=sr)

    # t, y = mw.create_1d_wavelet_from_scale(s=0.1)
    # log.info(f"energy = {MorletWavelet.calc_energy(y)}")
    # plt.plot(t, y, label="y_norm")
    # plt.show()
    # exit()

    # _, _, wavelet = mw.create_2d_wavelet_from_scale(s_f=0.02, s_t=0.01, reflect=False, normalize=False)
    # print(MorletWavelet.calc_energy(wavelet))
    # wavelet = wavelet.real.detach().numpy()
    # plt.imshow(wavelet)
    # plt.show()
    # exit()

    J_1 = 5
    Q_1 = 12
    # highest_freq = None
    # highest_freq = 20000
    highest_freq = 880
    # J_1 = 3   # No. of octaves
    # Q_1 = 16  # Steps per octave
    # highest_freq = 1760
    wavelet_bank, freqs = make_wavelet_bank(mw, J_1, Q_1, highest_freq_t=highest_freq)

    log.info(f"in audio.shape = {audio.shape}")
    # scalogram = calc_scalogram_td(audio, wavelet_bank, take_modulus=True)
    scalogram = calc_scalogram_fd(audio, wavelet_bank, take_modulus=True)
    log.info(f"scalogram shape = {scalogram.shape}")
    log.info(f"scalogram mean = {tr.mean(scalogram)}")
    log.info(f"scalogram std = {tr.std(scalogram)}")
    log.info(f"scalogram max = {tr.max(scalogram)}")
    log.info(f"scalogram min = {tr.min(scalogram)}")
    plot_scalogram(scalogram[0], title="chirp", dt=mw.dt, freqs=freqs)
    # plot_scalogram(scalogram[1], title="flute", dt=mw.dt, freqs=freqs)
