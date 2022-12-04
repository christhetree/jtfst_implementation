import logging
import os
from typing import Optional, List, Union, Tuple

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

    def y_2d(self, t_1: T, t_2: T, s: float = 1.0) -> T:
        assert t_1.ndim == 1
        assert t_2.ndim == 1
        assert t_1.size(0) * t_2.size(0) <= 2 ** 26  # TODO(cm)

        x_1 = t_1 / s
        y = self.a * (tr.exp(1j * self.w * x_1) - self.b)

        a = x_1 ** 2
        a = a.view(-1, 1).expand(-1, t_2.size(0))
        x_2 = t_2 / s
        b = x_2 ** 2
        b = b.view(1, -1).expand(t_1.size(0), -1)
        gauss = tr.exp(-0.5 * (a + b))

        y = y.view(-1, 1).expand(-1, gauss.size(1))
        y = y * gauss

        if y.dtype != self.dtype:
            y = y.to(self.dtype)
        return y

    def y_2d_op(self, t_1: T, t_2: T, s_1: float = 1.0, s_2: float = 1.0) -> T:
        assert t_1.ndim == 1
        assert t_2.ndim == 1
        assert t_1.size(0) * t_2.size(0) <= 2 ** 26  # TODO(cm)
        y_1 = MorletWavelet.normalize_to_unit_energy(self.y_1d(t_1, s_1))
        y_2 = MorletWavelet.normalize_to_unit_energy(self.y_1d(t_2, s_2))
        y = tr.outer(y_1, y_2)
        return y

    def create_2d_wavelet_from_scale(self, s: float = 1.0, normalize: bool = True) -> (T, T, T):
        with tr.no_grad():
            assert s >= self.min_scale
            M = int((self.n_sig * s) / self.dt)
            t_1 = tr.arange(-M, M + 1) * self.dt
            t_2 = tr.clone(t_1)
            wavelet = self.y_2d(t_1, t_2, s)
            if normalize:
                wavelet = MorletWavelet.normalize_to_unit_energy(wavelet)
            return t_1, t_2, wavelet

    def create_2d_wavelet_from_scale_op(self, s_1: float = 1.0, s_2: float = 1.0, normalize: bool = True) -> (T, T, T):
        with tr.no_grad():
            assert s_1 >= self.min_scale
            assert s_2 >= self.min_scale
            M_1 = int((self.n_sig * s_1) / self.dt)
            t_1 = tr.arange(-M_1, M_1 + 1) * self.dt
            M_2 = int((self.n_sig * s_2) / self.dt)
            t_2 = tr.arange(-M_2, M_2 + 1) * self.dt
            wavelet = self.y_2d_op(t_1, t_2, s_1, s_2)
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
    assert steps_per_octave >= 0

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
                      steps_per_octave_2: int = 0,
                      highest_freq_1: Optional[float] = None,
                      highest_freq_2: Optional[float] = None,
                      normalize: bool = True) -> (List[T], List[Union[Tuple[float, float], float]]):
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
                _, _, wavelet = mw.create_2d_wavelet_from_scale_op(s_1, s_2, normalize=normalize)
                wavelet_bank.append(wavelet)
                freqs.append((freq_1, freq_2))
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
    audio_fd = tr.fft.fft(audio, norm="backward").unsqueeze(1)

    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 1
        left_padding = max_padding - wavelet.size(-1) // 2
        right_padding = audio_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, -1).expand(-1, n_ch, -1)
        kernel = F.pad(kernel, (left_padding, right_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=0).unsqueeze(0)
    kernels_fd = tr.fft.ifft(kernels, norm="backward")
    out_fd = kernels_fd * audio_fd
    scalogram = tr.fft.ifft(out_fd, norm="forward")
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    scalogram = scalogram[:, :, :, :-max_padding]
    scalogram = tr.sum(scalogram, dim=2, keepdim=False)

    if take_modulus:
        scalogram = tr.abs(scalogram)

    return scalogram


def calc_jtfst_td(scalogram: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert scalogram.ndim == 3

    scalogram = scalogram.unsqueeze(1)  # Image with 1 channel
    scalogram_complex = scalogram.to(mw.dtype)
    convs = []
    for wavelet in tqdm(wavelet_bank):
        assert wavelet.ndim == 2
        kernel = wavelet.view(1, 1, *wavelet.shape)
        out = F.conv2d(scalogram_complex, kernel, stride=(1,), padding="same")
        convs.append(out)

    jtfst = tr.cat(convs, dim=1)
    if take_modulus:
        jtfst = tr.abs(jtfst)

    return jtfst


def calc_jtfst_fd(scalogram: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert scalogram.ndim == 3

    max_f_dim = max([w.size(0) for w in wavelet_bank])
    max_t_dim = max([w.size(1) for w in wavelet_bank])
    max_f_padding = max_f_dim // 2
    max_t_padding = max_t_dim // 2
    # TODO(cm): check why we can get away with only padding the front
    scalogram = F.pad(scalogram, (max_t_padding, 0, max_f_padding, 0))
    scalogram_fd = tr.fft.fft2(scalogram, norm="backward").unsqueeze(1)

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
    kernels_fd = tr.fft.ifft2(kernels, norm="backward")
    out_fd = kernels_fd * scalogram_fd
    jtfst = tr.fft.ifft2(out_fd, norm="forward")
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    jtfst = jtfst[:, :, :-max_f_padding, :-max_t_padding]

    if take_modulus:
        jtfst = tr.abs(jtfst)

    return jtfst



if __name__ == "__main__":
    n_samples = 24000
    # n_samples = 4 * 48000

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
    # chirp_audio = chirp_audio[:, :n_samples]
    chirp_audio = chirp_audio[:, -n_samples:-8000]
    chirp_audio = chirp_audio.view(1, 1, -1)
    assert audio_sr_1 == audio_sr_2

    # audio = flute_audio
    audio = chirp_audio
    # audio = tr.cat([flute_audio, chirp_audio], dim=0)
    # audio = tr.cat([tr.rand_like(flute_audio), flute_audio], dim=1)

    sr = audio_sr_1
    w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
    log.info(f"w = {w}")
    mw = MorletWavelet(w=w, sr=sr)

    # _, _, wavelet = mw.create_2d_wavelet_from_scale(s=0.02)
    # _, _, wavelet = mw.create_2d_wavelet_from_scale_op(s_1=0.01, s_2=0.01, normalize=False)
    # print(MorletWavelet.calc_energy(wavelet))
    # wavelet = wavelet.real.detach().numpy()
    # plt.imshow(wavelet)
    # plt.show()
    # exit()

    # t, y = mw.create_1d_wavelet_from_scale(s=0.1)
    # log.info(f"energy = {MorletWavelet.calc_energy(y)}")
    # plt.plot(t, y, label="y_norm")
    # plt.show()
    # exit()

    J_1 = 2
    Q_1 = 16
    highest_freq = None
    # highest_freq = 20000
    # highest_freq = 14080
    # J_1 = 3   # No. of octaves
    # Q_1 = 16  # Steps per octave
    # highest_freq = 1760
    wavelet_bank, freqs = make_wavelet_bank(mw, J_1, Q_1, highest_freq_1=highest_freq)

    log.info(f"in audio.shape = {audio.shape}")
    # scalogram = calc_scalogram_td(audio, wavelet_bank, take_modulus=True)
    scalogram = calc_scalogram_fd(audio, wavelet_bank, take_modulus=True)
    log.info(f"scalogram shape = {scalogram.shape}")
    log.info(f"scalogram mean = {tr.mean(scalogram)}")
    log.info(f"scalogram std = {tr.std(scalogram)}")
    log.info(f"scalogram max = {tr.max(scalogram)}")
    log.info(f"scalogram min = {tr.min(scalogram)}")
    # plot_scalogram(scalogram[0], title="flute", dt=mw.dt, freqs=freqs)
    # plot_scalogram(scalogram[1], title="chirp", dt=mw.dt, freqs=freqs)
    # exit()

    J_2_f = 0
    Q_2_f = 0
    highest_freq_f = None
    # highest_freq_f = 20000
    J_2_t = 0
    Q_2_t = 0
    highest_freq_t = None
    # highest_freq_t = 15000

    wavelet_bank_2, freqs_2 = make_wavelet_bank(mw, J_2_f, Q_2_f, J_2_t, Q_2_t, highest_freq_1=highest_freq_f, highest_freq_2=highest_freq_t)
    # for wavelet in wavelet_bank_2:
    #     wavelet = wavelet.real.detach().numpy()
    #     plt.imshow(wavelet)
    #     plt.show()

    log.info(f"in scalogram.shape = {scalogram.shape}")
    # jtfst = calc_jtfst_td(scalogram, wavelet_bank_2)
    jtfst = calc_jtfst_fd(scalogram, wavelet_bank_2)
    log.info(f"jtfst shape = {jtfst.shape}")
    log.info(f"jtfst mean = {tr.mean(jtfst)}")
    log.info(f"jtfst std = {tr.std(jtfst)}")
    log.info(f"jtfst max = {tr.max(jtfst)}")
    log.info(f"jtfst min = {tr.min(jtfst)}")
    real = jtfst[0, 0, :, :].squeeze().real.detach().numpy()
    plt.imshow(real, aspect="auto", interpolation="none", cmap="OrRd")
    plt.show()
