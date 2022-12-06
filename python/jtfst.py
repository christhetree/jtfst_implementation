import logging
import os

import torch as tr
import torchaudio

from scalogram_1d import plot_scalogram_1d, calc_scalogram_1d
from wavelets import MorletWavelet, make_wavelet_bank

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


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
    # scalogram = calc_scalogram_1d_td(audio, wavelet_bank, take_modulus=True)
    scalogram = calc_scalogram_1d(audio, wavelet_bank, take_modulus=True)
    log.info(f"scalogram shape = {scalogram.shape}")
    log.info(f"scalogram mean = {tr.mean(scalogram)}")
    log.info(f"scalogram std = {tr.std(scalogram)}")
    log.info(f"scalogram max = {tr.max(scalogram)}")
    log.info(f"scalogram min = {tr.min(scalogram)}")
    plot_scalogram_1d(scalogram[0], title="chirp", dt=mw.dt, freqs=freqs)
    # plot_scalogram_1d(scalogram[1], title="flute", dt=mw.dt, freqs=freqs)
