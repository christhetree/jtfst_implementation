"""
Preprocess features by concatenating them into a single matrix and
creating binary labels indicating whether a playing technique is present.

Also calculates the mean, standard deviation, and first order difference
over a sliding window comprising five freames.
"""
import logging
import os
import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
import scipy.io

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Default temporal support for each technique
temporal_support = {
    "acciacatura": 12,
    "portamento": 15,
    "glissando": 14,
}


def load_features(input_file: Path):
    log.info(f"Loading features from {input_file}")
    if input_file.suffix == ".mat":
        features = scipy.io.loadmat(input_file)["fileFeatures"][0, :]
    else:
        # TODO: Add support for npy files?
        raise ValueError(f"Unknown file type: {input_file}")

    return features


def load_wavfile_list(filelist: Path):
    with open(filelist, "r") as f:
        wavfiles = [Path(l.strip()) for l in f.readlines()]
    return wavfiles


def get_annotation_filelist(wav_files: List[Path], tech_name: str):
    """Get annotations for each file"""
    anno_files = []
    for k in range(len(wav_files)):
        if wav_files[k].parts[2] == "Iso":
            anno_files.append(wav_files[k].with_suffix(".csv"))
        elif wav_files[k].parts[2] == "Piece":  # 'Piece'
            anno_techname = wav_files[k].with_name(
                wav_files[k].stem + "_tech_" + tech_name + ".csv"
            )
            if anno_techname.exists():
                anno_files.append(anno_techname)
            else:
                anno_files.append(None)

    return anno_files


def temporal_summarization(features, context: int = 2) -> np.ndarray:
    """
    Calculate the mean, standard deviation, and first order difference
    over a temporal frames equal to context*2 + 1
    """
    for i in range(len(features)):
        x = np.copy(features[i])

        # Repeat the jtfst dimension for statistics
        x = np.tile(x, (3, 1))
        jtfs_len = features[i].shape[0]

        for n in range(context, x.shape[1] - context):
            x[:jtfs_len, n] = np.mean(
                x[:jtfs_len, n - context : n + context + 1], axis=1
            )
            x[jtfs_len : jtfs_len * 2, n] = np.std(
                x[jtfs_len : jtfs_len * 2, n - context : n + context + 1], axis=1
            )
            # Not totally clear how to calculate the first order difference across a frame of 5 samples
            # and then store it in a single time point?
            x[jtfs_len * 2 :, n] = np.mean(
                np.diff(x[jtfs_len * 2 :, n - context : n + context + 1], axis=1),
                axis=1,
            )

        features[i] = x

    return features


def get_player_id(filename):
    """
    Get player ID from filename
    """
    return int(filename.parts[1].replace("Player", ""))


def concatenate_features(features, wavfiles):
    """
    Concatenate all file features into single matrix.
    """
    # Create arrays of the player ID and file ID for each concatenated feature
    player_ids = []
    file_ids = []
    for i, f in enumerate(features):
        player_ids.append(np.ones(f.shape[1], dtype=int) * get_player_id(wavfiles[i]))
        file_ids.append(np.ones(f.shape[1], dtype=int) * i)

    player_ids = np.concatenate(player_ids)
    file_ids = np.concatenate(file_ids)

    # Concatenate all temporal features into a single matrix
    features = np.concatenate(features, axis=1)
    features = np.transpose(features)

    assert len(features) == len(file_ids)
    assert len(features) == len(player_ids)

    return features, player_ids, file_ids


def get_feature_labels(
    annotations: List[Path],
    features: np.ndarray,
    file_ids: np.ndarray,
    tech_name: str,
    samplerate: int,
    t: int,
    oversampling: int,
):
    """
    Create binary labels indicating whether a playing technique is present in a given frame.
    """
    # Scattering hop size
    if t is None:
        t = temporal_support[tech_name.lower()]

    hop_size = (2**t) / (2**oversampling)
    log.info("Hop size: %d ms", hop_size / samplerate * 1000)

    labels = []
    for i, a in enumerate(annotations):
        x = np.zeros_like(np.where(file_ids == i)[0])
        if a is not None and a.stem.split("_")[-1].lower() == tech_name.lower():
            # Load annotations
            file_anno = pd.read_csv(a)
            file_onoff = np.hstack(
                (float(list(file_anno)[0]), file_anno[list(file_anno)[0]])
            )

            # Set labels during PET to 1
            prev_start = 0
            prev_end = 0
            overlapping_events = 0
            for n in range(len(file_onoff) // 2):
                start_idx = int(file_onoff[2 * n] * samplerate / hop_size)
                end_idx = int(file_onoff[2 * n + 1] * samplerate / hop_size)

                # Handle an edge case where the end index is the same or less than the start index
                end_idx = end_idx if end_idx > start_idx else start_idx + 1

                assert start_idx < len(
                    x
                ), f"Start index {start_idx} is larger than the number of frames {len(x)} in {a}"
                assert end_idx < len(
                    x
                ), f"End index {end_idx} is larger than the number of frames {len(x)} in {a}"

                if start_idx == prev_start and end_idx == prev_end:
                    log.info(f"Repeated event in {a}")
                    continue

                # Set label to 1 for all frames during PET
                x[start_idx:end_idx] = 1

                # Check for overlapping events
                if start_idx <= prev_end:
                    log.info(f"Overlapping events in {a}")
                    overlapping_events += 1

                prev_start = start_idx
                prev_end = end_idx

            # Make sure enough events are present -- the number of
            # changes in the label vector (i.e. event on: 0->1)
            # should be equal to half the number of events in the annotation file
            # minus the number of overlapping events
            expected_events = (len(file_onoff) // 2) - overlapping_events
            found_events = len(np.where(np.diff(np.pad(x, (1, 1))) == 1)[0])
            if expected_events != found_events:
                print(x)
            assert (
                expected_events == found_events
            ), f"Found {expected_events} events in {a} but {found_events} in the label vector"

        labels.append(x)

    return np.concatenate(labels)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        help="Input features to be processed",
        type=str,
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
        help="Output to save features to -- defaults to features/<techname>.npz",
    )
    parser.add_argument(
        "--samplerate",
        default=44100,
        help="Sample rate that features were extracted at",
    )
    parser.add_argument(
        "-t",
        default=None,
        help="Exponent of power of two of T used in dJTFST",
    )
    parser.add_argument(
        "--oversampling",
        default=2,
        help="Oversampling factor for dJTFST",
    )

    args = parser.parse_args(arguments)
    features = load_features(Path(args.input))
    log.info(f"Loaded features for {len(features)} files")

    # Create output directory if it doesn't exist
    if args.outname is None:
        args.outname = Path("features") / f"{args.techname.lower()}.npz"

    # Create the output directory if it doesn't exist
    if not args.outname.parent.exists():
        Path(args.outname.parent).mkdir(parents=True)

    # If the output file already exists, raise an error
    if args.outname.exists():
        raise ValueError(f"Output file {args.outname} already exists")

    # Load wavefile and annotation file lists
    wavfiles = load_wavfile_list(Path(args.filelist))
    assert len(features) == len(wavfiles)

    annos_file = get_annotation_filelist(wavfiles, args.techname)
    assert len(features) == len(annos_file)

    # Procress the features by concatenating them into a single matrix and
    # computing the mean, stdev, and first order difference
    features = temporal_summarization(features)
    features, player_ids, file_ids = concatenate_features(features, wavfiles)
    log.info(
        f"Concenated all file features into single marix of size: {features.shape}"
    )

    labels = get_feature_labels(
        annos_file,
        features,
        file_ids,
        args.techname,
        args.samplerate,
        args.t,
        args.oversampling,
    )
    assert len(features) == len(labels)

    # Save the features and labels to disk
    np.savez(
        args.outname,
        features=features,
        labels=labels,
        player_ids=player_ids,
        file_ids=file_ids,
    )


if __name__ == "__main__":
    log.info("Feature Preprocessing")
    sys.exit(main(sys.argv[1:]))
