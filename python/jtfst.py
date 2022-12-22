import logging
import os
from typing import Optional, List, Tuple

import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T
from tqdm import tqdm

from scalogram_1d import plot_scalogram_1d
from scattering_1d import calc_scat_transform_1d, calc_scat_transform_1d_jagged, average_td
from scattering_2d import calc_scat_transform_2d_fast
from signals import make_pure_sine, make_pulse, make_exp_chirp
from wavelets import DiscreteWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_jtfst_1d(x: T,
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
                  avg_win_t: Optional[int] = 2 ** 12) -> (T, List[float], T, List[Tuple[float, float, int]]):
    log.info(f"in x.shape = {x.shape}")
    scalogram, freqs_1, _ = calc_scat_transform_1d(x,
                                                   sr,
                                                   J_1,
                                                   Q_1,
                                                   should_avg=False)
    log.info(f"scalogram shape = {scalogram.shape}")
    y_t, freqs_t = calc_scat_transform_1d_jagged(scalogram,
                                                 sr,
                                                 freqs_1,
                                                 J_2_t,
                                                 Q_2_t,
                                                 should_avg=False,
                                                 should_pad=True)
    freqs_2 = []
    jtfst_s = []
    for y, freq_t in tqdm(zip(y_t, freqs_t)):
        y = y.squeeze(1)
        y = tr.swapaxes(y, 1, 2)
        jtfst, freqs_f, _ = calc_scat_transform_1d(y,
                                                   sr,
                                                   J_2_f,
                                                   Q_2_f,
                                                   should_avg=False,
                                                   squeeze_channels=False,
                                                   reflect_t=False)  # TODO(cm): why is this not changing anything
        jtfst = tr.swapaxes(jtfst, 2, 3)
        jtfst_s.append(jtfst)
        for freq_f in freqs_f:
            freqs_2.append((freq_f, freq_t, 1))

    jtfst = tr.cat(jtfst_s, dim=1)
    if should_avg_t:
        if avg_win_t is None:
            lowest_freq_t = freqs_t[-1]
            assert sr % lowest_freq_t == 0
            avg_win_t = int(sr / lowest_freq_t)

        log.info(f"avg_win_t = {avg_win_t}")
        jtfst = average_td(jtfst, avg_win_t, dim=-1)

    if should_avg_f:
        if avg_win_f is None:
            log.warning(f"should_avg_f is True, but avg_win_f is None, using a heuristic value of 2")
            avg_win_f = 2

        log.info(f"avg_win_f = {avg_win_f}")
        jtfst = average_td(jtfst, avg_win_f, dim=-2)

    log.info(f"jtfst shape = {jtfst.shape}")
    return scalogram, freqs_1, jtfst, freqs_2


def calc_jtfst_2d(x: T,
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
                  avg_win_t: Optional[int] = 2 ** 12) -> (T, List[float], T, List[Tuple[float, float, int]]):
    log.info(f"in x.shape = {x.shape}")
    scalogram, freqs_1, _ = calc_scat_transform_1d(x,
                                                   sr,
                                                   J_1,
                                                   Q_1,
                                                   should_avg=False)
    log.info(f"scalogram shape = {scalogram.shape}")
    jtfst, freqs_2 = calc_scat_transform_2d_fast(scalogram,
                                                 sr,
                                                 J_2_f,
                                                 J_2_t,
                                                 Q_2_f,
                                                 Q_2_t,
                                                 should_avg_f,
                                                 should_avg_t,
                                                 avg_win_f,
                                                 avg_win_t)
    log.info(f"jtfst shape = {jtfst.shape}")
    return scalogram, freqs_1, jtfst, freqs_2


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

    J_1 = 12
    J_2_f = 2
    J_2_t = 12
    Q_1 = 16
    Q_2_f = 1
    Q_2_t = 1
    should_avg_f = True
    should_avg_t = True
    avg_win_f = 4  # Average across 25% of an octave if Q_1 == 16
    avg_win_t = 2 ** J_1

    # jtfst_func = calc_jtfst_1d
    jtfst_func = calc_jtfst_2d

    scalogram, freqs_1, jtfst, freqs_2 = jtfst_func(audio,
                                                    sr,
                                                    J_1=J_1,
                                                    J_2_f=J_2_f,
                                                    J_2_t=J_2_t,
                                                    Q_1=Q_1,
                                                    Q_2_f=Q_2_f,
                                                    Q_2_t=Q_2_t,
                                                    should_avg_f=should_avg_f,
                                                    should_avg_t=should_avg_t,
                                                    avg_win_f=avg_win_f,
                                                    avg_win_t=avg_win_t)
    plot_scalogram_1d(scalogram[0], title="scalo", dt=None, freqs=freqs_1, n_y_ticks=12)

    pic_idx = -2
    log.info(f"jtfst shape = {jtfst.shape}")
    log.info(f"jtfst energy = {DiscreteWavelet.calc_energy(jtfst)}")
    mean = tr.mean(jtfst)
    std = tr.std(jtfst)
    jtfst = tr.clip(jtfst, mean - (4 * std), mean + (4 * std))
    pic = jtfst[0, pic_idx, :, :].detach().numpy()
    plt.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
    plt.title("jtfst")
    plt.show()
    exit()

    # Plotting
    n_rows = len(freqs_2) // 2
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx, (freq_f, freq_t, theta) in enumerate(freqs_2):
        curr_ax = ax[idx // 2, idx % 2]
        pic = jtfst[0, idx, :, :].squeeze().detach().numpy()
        curr_ax.imshow(pic, aspect="auto", interpolation="none", cmap="OrRd")
        curr_ax.set_title(f"freq_f = {freq_f:.0f}, freq_t = {freq_t:.0f}, theta = {theta}")
    plt.show()
