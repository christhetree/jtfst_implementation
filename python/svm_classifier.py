import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# classifier-SVM hyperparameters and grid search
kernel = "rbf"
gpu_id = 0
param_grid = {
    "C": [256, 128, 64, 32, 16, 8],
    "gamma": [2 ** (-12), 2 ** (-11), 2 ** (-10), 2 ** (-9), 2 ** (-8), 2 ** (-7)],
}  # para_grid used
scoring = "f1_macro"
cv = 3


def load_dataset(input_file: str):
    log.info(f"Loading dataset from {input_file}")
    dataset = np.load(input_file)
    assert "features" in dataset.files
    assert "labels" in dataset.files
    assert "player_ids" in dataset.files
    assert "file_ids" in dataset.files
    return dataset


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        help="Input dataset to be used for classification",
        type=str,
    )

    args = parser.parse_args(arguments)

    # Load the dataset
    dataset = load_dataset(args.input)


if __name__ == "__main__":
    log.info("SVM classifier")
    sys.exit(main(sys.argv[1:]))
