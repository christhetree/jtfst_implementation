import logging
import os
import argparse
from pathlib import Path
import sys

import numpy as np
import scipy.io

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def load_features(input_file: Path):
    log.info(f"Loading features from {input_file}")
    if input_file.suffix == ".mat":
        features = scipy.io.loadmat(input_file)["fileFeatures"][0, :]
    else:
        # TODO: Add support for npy files?
        raise ValueError(f"Unknown file type: {input_file}")

    return features


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


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        help="Input file to be processed",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="./features",
        help="Output folder for preprocessed features",
    )

    args = parser.parse_args(arguments)
    features = load_features(Path(args.input))
    log.info(f"Loaded features for {len(features)} files")

    features = temporal_summarization(features)
    log.info(f"{features.shape}")


if __name__ == "__main__":
    log.info("Feature Preprocessing")
    sys.exit(main(sys.argv[1:]))
