import argparse
import logging
import math
import os
from pathlib import Path
import sys
import time

import pickle
import numpy as np
import torch as tr
import torchaudio
from torch import Tensor as T
from tqdm import tqdm
from typing import List

from jtfst import JTFST2D

# Maximum time modulation to use in Hz
TIME_MOD_MAX = 50.0
params = {
    "acciacatura": {
        "J_1": 12,
        "J_2_f": 2,
        "J_2_t": 12,
        "Q_1": 12,
        "Q_2_f": 2,
        "Q_2_t": 1,
        "should_avg_f": True,
        "should_avg_t": True,
        "avg_win_f": 2,  # Average across 25% of an octave
        "avg_win_t": 2 ** 10,
        "reflect_f": True
    },
    "glissando": {
        "J_1": 12,
        "J_2_f": 2,
        "J_2_t": 12,
        "Q_1": 12,
        "Q_2_f": 2,
        "Q_2_t": 1,
        "should_avg_f": True,
        "should_avg_t": True,
        "avg_win_f": 2,  # Average across 25% of an octave
        "avg_win_t": 2 ** 12,
        "reflect_f": True
    },
    "portamento":
    {
        "J_1": 12,
        "J_2_f": 2,
        "J_2_t": 12,
        "Q_1": 16,
        "Q_2_f": 2,
        "Q_2_t": 1,
        "should_avg_f": True,
        "should_avg_t": True,
        "avg_win_f": 2,  # Average across 25% of an octave
        "avg_win_t": 2 ** 13,
        "reflect_f": True
    }
}


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

tr.backends.cudnn.benchmark = True
log.info(f'tr.backends.cudnn.benchmark = {tr.backends.cudnn.benchmark}')

GPU_IDX = 0

if tr.cuda.is_available():
    log.info(f"tr.cuda.device_count() = {tr.cuda.device_count()}")
    tr.cuda.set_device(GPU_IDX)
    log.info(f"tr.cuda.current_device() = {tr.cuda.current_device()}")
    device = tr.device(f"cuda:{GPU_IDX}")
else:
    log.info(f"No GPUs found")
    device = tr.device("cpu")

log.info(f"device = {device}")

def load_wavfile_list(filelist: Path):
    with open(filelist, "r") as f:
        wavfiles = [Path(l.strip()) for l in f.readlines()]
    return wavfiles

def load_audio(path: str) -> (T, int):
    audio, sr = torchaudio.load(path)
    audio = tr.mean(audio, dim=0)
    return audio, sr

def extract_files(files: List[Path], tech_name: str):
    features = []
    for f in tqdm(files):
        audio, sr = load_audio(f)
        file_features = extract_audio(audio, sr, tech_name)
        print(file_features.shape)
        features.append(file_features.numpy())

    return features

def dimension_reduction(features: T, freqs: List[T], average_orientation=True):

    keep_dims = []
    for i in range(len(freqs)):
        if freqs[i][1] <= TIME_MOD_MAX:
            keep_dims.append(i)
    
    # Only keep the joint features that capture
    # time modulations the correct modulation rate
    features = features[:, keep_dims, :, :]

    # Average across the orientation dimension
    if average_orientation:
        f = features.shape
        features = features.reshape(f[0], f[1] // 2, 2, f[2], f[3])
        features = features.mean(dim=2)

    return features


def extract_audio(audio: T, sr: int, tech_name: str):
    jtfst_model = JTFST2D(sr, **params[tech_name])
    jtfst_model.to(device)

    # Split audio into multiples of the averaging window size close to 1sec
    frame_size = (sr // params[tech_name]["avg_win_t"]) * params[tech_name]["avg_win_t"]
    split_indices = tr.arange(audio.shape[0])[::frame_size][1:]
    audio = list(tr.tensor_split(audio, split_indices))

    # Pad last frame up to a multiple of the averaging window
    remainder = audio[-1].shape[0] % params[tech_name]["avg_win_t"]
    if remainder > 0:
        audio[-1] = tr.nn.functional.pad(audio[-1], (0, params[tech_name]["avg_win_t"] - remainder))

    file_features = []
    for frame in audio:
        # Add dims for batch and stereo
        batch = frame.repeat(1, 1, 1)
        batch = batch.to(device)

        # Compute JTFST
        scalogram, freqs_1, jtfst, freqs_2 = jtfst_model(batch)
        jtfst = dimension_reduction(jtfst, freqs_2, average_orientation=params[tech_name]["reflect_f"])
        jtfst = jtfst.flatten(start_dim=1, end_dim=2)
        jtfst = jtfst.squeeze(0)
        file_features.append(jtfst.cpu())

    file_features = tr.cat(file_features, dim=-1)
    return file_features

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "filelist",
        help="Text file with ordering of files in the feature array",
        type=str,
    )
    parser.add_argument(
        "techname",
        help="Technique name [acciacatura, portamento, glissando]",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outname",
        default=None,
        help="Output to save features to -- defaults to features/jtfst_<techname>.pkl",
    )

    args = parser.parse_args(arguments)

    # Create output directory if it doesn't exist
    if args.outname is None:
        args.outname = Path("features") / f"jtfst_{args.techname.lower()}.pkl"

    # Create the output directory if it doesn't exist
    if not args.outname.parent.exists():
        Path(args.outname.parent).mkdir(parents=True)

    # If the output file already exists, raise an error
    if args.outname.exists():
        raise ValueError(f"Output file {args.outname} already exists")

    # Load wavefiles
    wavfiles = load_wavfile_list(Path(args.filelist))
    features = extract_files(wavfiles, args.techname)
    
    # Save the extracted features
    with open(args.outname, "wb") as fp:
        pickle.dump(features, fp)

if __name__ == "__main__":
    log.info("Feature Extraction")
    sys.exit(main(sys.argv[1:]))
