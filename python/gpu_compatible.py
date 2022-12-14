import logging
import os
from typing import Optional

import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T, nn
from tqdm import tqdm

from filterbanks import make_wavelet_bank
from scalogram_1d import plot_scalogram_1d
from scattering_1d import _calc_scat_transform_1d, calc_scat_transform_1d_jagged, average_td
from scattering_2d import calc_scat_transform_2d_fast
from signals import make_pure_sine, make_pulse, make_exp_chirp
from wavelets import MorletWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

tr.backends.cudnn.benchmark = True
log.info(f'tr.backends.cudnn.benchmark = {tr.backends.cudnn.benchmark}')

GPU_IDX = 1

if tr.cuda.is_available():
    log.info(f"tr.cuda.device_count() = {tr.cuda.device_count()}")
    tr.cuda.set_device(GPU_IDX)
    log.info(f"tr.cuda.current_device() = {tr.cuda.current_device()}")
    device = tr.device(f"cuda:{GPU_IDX}")
else:
    log.info(f"No GPUs found")
    device = tr.device("cpu")

log.info(f"device = {device}")


class ScatTransform1D(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int = 12,
                 Q: int = 16,
                 should_avg: bool = False,
                 highest_freq: Optional[float] = None,
                 avg_win: Optional[int] = None,
                 squeeze_channels: bool = True,
                 reflect_t: bool = False) -> None:
        super().__init__()
        self.sr = sr
        self.J = J
        self.Q = Q
        self.should_avg = should_avg
        self.highest_freq = highest_freq
        self.avg_win = avg_win
        self.squeeze_channels = squeeze_channels
        self.reflect_t = reflect_t

        w = MorletWavelet.freq_to_w_at_s(1.0, s=1.0)
        mw = MorletWavelet(w=w, sr_t=sr)
        wavelet_bank, _, freqs_t, _ = make_wavelet_bank(mw, J, Q, highest_freq, reflect_t=reflect_t)
        self.wavelet_bank = nn.ParameterList(wavelet_bank)
        self.freqs_t = freqs_t

    def forward(self, x: T) -> T:
        with tr.no_grad():
            y = _calc_scat_transform_1d(
                x, self.sr, self.wavelet_bank, self.freqs_t, self.should_avg, self.avg_win, self.squeeze_channels)
            return y


class JTFST1D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_1: int = 12,
                 J_2_f: int = 2,
                 J_2_t: int = 12,
                 Q_1: int = 16,
                 Q_2_f: int = 1,
                 Q_2_t: int = 1,
                 should_avg_f: bool = True,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = 2 ** 2,
                 avg_win_t: Optional[int] = 2 ** 12) -> None:
        super().__init__()
        self.scat_1d_1 = ScatTransform1D(sr, J_1, Q_1, should_avg=False)
        self.scat_1d_3 = ScatTransform1D(sr,
                                         J_2_f,
                                         Q_2_f,
                                         should_avg=False,
                                         squeeze_channels=False,
                                         reflect_t=False)   # TODO(cm): should be true eventually
        self.sr = sr
        self.J_2_t = J_2_t
        self.Q_2_t = Q_2_t
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t

    def forward(self, x: T) -> T:
        # TODO(cm): tmp duplicate code
        with tr.no_grad():
            scalogram = self.scat_1d_1(x)
            y_t, freqs_t = calc_scat_transform_1d_jagged(scalogram,
                                                         self.sr,
                                                         self.scat_1d_1.freqs_t,
                                                         self.J_2_t,
                                                         self.Q_2_t,
                                                         should_avg=False,
                                                         should_pad=True)
            freqs_2 = []
            jtfst_s = []
            for y, freq_t in zip(y_t, freqs_t):
                y = y.squeeze(1)
                y = tr.swapaxes(y, 1, 2)
                jtfst = self.scat_1d_3(y)
                jtfst = tr.swapaxes(jtfst, 2, 3)
                jtfst_s.append(jtfst)
                for freq_f in self.scat_1d_3.freqs_t:
                    freqs_2.append((freq_f, freq_t, 1))

            jtfst = tr.cat(jtfst_s, dim=1)
            if self.should_avg_t:
                if self.avg_win_t is None:
                    lowest_freq_t = freqs_t[-1]
                    assert self.sr % lowest_freq_t == 0
                    avg_win_t = int(self.sr / lowest_freq_t)
                else:
                    avg_win_t = self.avg_win_t
                jtfst = average_td(jtfst, avg_win_t, dim=-1)

            if self.should_avg_f:
                if self.avg_win_f is None:
                    log.warning(f"should_avg_f is True, but avg_win_f is None, using a heuristic value of 2")
                    avg_win_f = 2
                else:
                    avg_win_f = self.avg_win_f
                jtfst = average_td(jtfst, avg_win_f, dim=-2)

            return jtfst


class JTFST2D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_1: int = 12,
                 J_2_f: int = 2,
                 J_2_t: int = 12,
                 Q_1: int = 16,
                 Q_2_f: int = 1,
                 Q_2_t: int = 1,
                 should_avg_f: bool = True,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = 2 ** 2,
                 avg_win_t: Optional[int] = 2 ** 12) -> None:
        super().__init__()
        self.scat_1d = ScatTransform1D(sr, J_1, Q_1, should_avg=False)
        self.sr = sr
        self.J_2_f = J_2_f
        self.J_2_t = J_2_t
        self.Q_2_f = Q_2_f
        self.Q_2_t = Q_2_t
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t

    def forward(self, x: T) -> T:
        with tr.no_grad():
            scalogram = self.scat_1d(x)
            jtfst, freqs_2 = calc_scat_transform_2d_fast(scalogram,
                                                         self.sr,
                                                         self.J_2_f,
                                                         self.J_2_t,
                                                         self.Q_2_f,
                                                         self.Q_2_t,
                                                         self.should_avg_f,
                                                         self.should_avg_t,
                                                         self.avg_win_f,
                                                         self.avg_win_t)
            return jtfst


if __name__ == "__main__":
    start_n = 0
    n_samples = 2 ** 16
    # n_samples = 4096

    audio_path = "../data/flute.wav"
    flute_audio, sr = torchaudio.load(audio_path)
    flute_audio = flute_audio[:, start_n:n_samples]
    flute_audio = tr.mean(flute_audio, dim=0)

    audio = flute_audio

    sr = 48000
    audio_1 = make_pure_sine(n_samples, sr, freq=4000, amp=1.0)
    audio_2 = make_pulse(n_samples, center_loc=0.5, dur_samples=128, amp=4.0)
    audio_3 = make_exp_chirp(n_samples, sr, start_freq=20, end_freq=20000, amp=1.0)
    audio = audio_1 + audio_2 + audio_3

    audio = audio.view(1, 1, -1)

    bs = 1
    n_iters = 100
    audio = audio.repeat(bs, 1, 1)
    # audio = tr.cat([audio, tr.rand_like(audio)], dim=0)

    audio = audio.to(device)
    log.info(f"audio.is_cuda = {audio.is_cuda}")

    st_1d = ScatTransform1D(sr)
    st_1d.to(device)

    jtfst_1d = JTFST1D(sr)
    jtfst_1d.to(device)

    jtfst_2d = JTFST2D(sr)
    jtfst_2d.to(device)

    log.info(f"in audio.shape = {audio.shape}")
    for _ in tqdm(range(n_iters)):
        scalogram = st_1d(audio)
        jtfst = jtfst_1d(audio)
        # jtfst = jtfst_2d(audio)

    batch_idx = 0
    log.info(f"scalogram shape = {scalogram.shape}")
    log.info(f"scalogram energy = {MorletWavelet.calc_energy(scalogram)}")
    mean = tr.mean(scalogram)
    std = tr.std(scalogram)
    scalogram_to_plot = tr.clip(scalogram, mean - (4 * std), mean + (4 * std))
    plot_scalogram_1d(scalogram_to_plot[batch_idx], title="scalo", dt=None, freqs=st_1d.freqs_t, n_y_ticks=12)

    pic_idx = -2
    log.info(f"jtfst shape = {jtfst.shape}")
    log.info(f"jtfst energy = {MorletWavelet.calc_energy(jtfst)}")
    mean = tr.mean(jtfst)
    std = tr.std(jtfst)
    jtfst = tr.clip(jtfst, mean - (4 * std), mean + (4 * std))
    pic = jtfst[batch_idx, pic_idx, :, :].detach().numpy()
    plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    plt.title("jtfst_gpu")
    plt.show()
    exit()
