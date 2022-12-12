import argparse
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# classifier-SVM hyperparameters and grid search
kernel = "rbf"
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


def test_train_split(dataset):
    # data split according to players + cross validation
    torch.manual_seed(42)
    player_id = dataset["player_ids"]
    player_split = torch.randperm(len(np.unique(player_id))) + 1
    player_split = player_split.numpy()

    # train_split test_split player
    train_split = []
    test_split = []

    train_split.append(player_split[0 : int(player_split.shape[0] * 0.8)])
    test_split.append(
        player_split[int(player_split.shape[0] * 0.8) : player_split.shape[0]]
    )

    train_split.append(player_split[2:10])
    test_split.append(player_split[0:2])

    train_split.append(np.hstack((player_split[4:10], player_split[0:2])))
    test_split.append(player_split[2:4])

    train_split.append(np.hstack((player_split[6:10], player_split[0:4])))
    test_split.append(player_split[4:6])

    train_split.append(np.hstack((player_split[8:10], player_split[0:6])))
    test_split.append(player_split[6:8])

    return train_split, test_split


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        help="Input dataset to be used for classification",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU to use for training, defaults to None",
        type=str,
    )

    args = parser.parse_args(arguments)

    dataset = load_dataset(args.input)
    train_split, test_split = test_train_split(dataset)


if __name__ == "__main__":
    log.info("SVM classifier")
    sys.exit(main(sys.argv[1:]))
