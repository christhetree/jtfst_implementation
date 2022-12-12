import argparse
import logging
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

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


def load_svm(gpu_id):
    try:
        from thundersvm import SVC
    except (ImportError, FileNotFoundError) as e:
        log.error(f"thundersvm error: {e}")
        if gpu_id is not None:
            log.error("thundersvm error and gpu_id is not None")
            sys.exit(1)
        else:
            from sklearn.svm import SVC

            log.info("Using sklearn.svm.SVC")
            return SVC

    log.info("Using thundersvm.SVC")
    return SVC


def get_outfile_name(input):
    # get output file name
    input = Path(input)
    filename = input.stem
    filename = filename + "_svm_results.npz"
    filename = Path("results") / filename

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)

    assert not filename.exists(), f"Output file {filename} already exists"
    log.info("Writing results to: {}".format(filename))
    return filename


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


def run(dataset, train_split, test_split, results_file, SVC, gpu_id):
    num_splits = len(train_split)

    # Load labels
    labels = dataset["labels"]
    num_classes = len(np.unique(labels))
    assert num_classes == 2, "Only binary classification is supported"

    results = []
    confusion = np.zeros((num_splits, num_classes, num_classes))

    features = dataset["features"]
    player_ids = dataset["player_ids"]

    for split, (train, test) in tqdm(enumerate(zip(train_split, test_split))):

        # Split data
        train_idx = np.isin(player_ids, train)
        test_idx = np.isin(player_ids, test)
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

        # There should be no NaNs in the data
        assert not np.isnan(train_features).any()
        assert not np.isnan(test_features).any()

        # Normalization
        stdscaler = StandardScaler()
        train_features = stdscaler.fit_transform(train_features)
        test_features = stdscaler.transform(test_features)
        log.info(f"Train: {train_features.shape}, Test: {test_features.shape}")

        svc = (
            SVC(kernel=kernel, gpu_id=gpu_id)
            if gpu_id is not None
            else SVC(kernel=kernel)
        )
        clf = GridSearchCV(svc, param_grid=param_grid, cv=cv, scoring=scoring)
        clf = clf.fit(train_features[:1000], train_labels[:1000])
        label_pred = clf.predict(test_features)

        print("Result of split %d :" % split)
        print(classification_report(test_labels, label_pred))
        print(confusion_matrix(test_labels, label_pred))

        results.append(classification_report(test_labels, label_pred, output_dict=True))
        confusion[split] = confusion_matrix(test_labels, label_pred)

    np.savez(results_file, results=results, confusion=confusion)


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
    output_filename = get_outfile_name(args.input)

    # Attempt to load thundersvm
    SVC = load_svm(args.gpu)

    dataset = load_dataset(args.input)
    train_split, test_split = test_train_split(dataset)
    run(dataset, train_split, test_split, output_filename, SVC, args.gpu)


if __name__ == "__main__":
    log.info("SVM classifier")
    sys.exit(main(sys.argv[1:]))
