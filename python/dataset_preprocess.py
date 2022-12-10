"""
Filters the CBF Dataset to only
include PETs (i.e. acciacatura, portamento, and glissando)
Segments audio files so all are less than 60 seconds.
"""
from copy import copy
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import soundfile

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# PETs to include
tech_types = ["Glissando", "Portamento", "Acciacatura"]


def get_annotation_files(wavfiles: List[Path]):
    """
    Load all the annotation files for each wavfile
    Check to make sure the wavfile contains annotations
    for a technique type that we are considering.
    """
    pets = []
    for w in wavfiles:
        # Get all the annotation files for the wavfile -- there may be more than one
        annotations = list(Path(w.parent).rglob(f"{w.stem}*.csv"))
        annos_for_wav = []
        for a in annotations:
            tech = a.stem.split("_")[-1]
            if tech in tech_types:
                annos_for_wav.append(a)

        if len(annos_for_wav) > 0:
            pets.append((w, annos_for_wav))

    return pets


def load_annotations(annotations: List[Path]):
    """
    Load all annotations for a file and store them in an interval tree.
    This will allow us to quickly check if a given timepoint is annotated
    """
    tree = IntervalTree()
    for a in annotations:
        tech = a.stem.split("_")[-1]
        annos = pd.read_csv(a, header=None, names=["t", "desc"])
        times = [t[1]["t"] for t in annos.iterrows()]
        # Each annotation is a pair of times: begin and end
        for k in range(len(times) // 2):
            begin = times[2 * k]
            end = times[2 * k + 1]
            tree.addi(begin, end, tech)
    return tree


def split_audio(
    audio: np.ndarray,
    sr: float,
    annotations: IntervalTree,
    segment_length: float = 30.0,
    previous_split: List = None,
):
    """
    Split an audio file and the corresponding annotations into segments
    that have a target length of segment_length seconds.
    This function is recursive and will continue to split the audio file
    until it has been split into segments that are all less than 2x segment_length.
    """
    annotations = annotations.copy()
    num_annotations = len(annotations)

    # Create new split list if this is the first pass
    splits = [] if previous_split is None else previous_split
    duration = len(audio) / sr

    if duration > segment_length * 2.0:

        target_end = segment_length

        # Make sure that target event doesn't land in the middle
        # of PET event -- if it does, then look for the next
        # gap between events greater than 0.5sec to split the file
        if len(annotations[target_end - 0.5 : target_end + 0.5]):
            a_iter = iter(sorted(annotations[target_end:]))
            prev_end = next(a_iter).end
            while True:
                next_event = next(a_iter)
                if next_event.begin - prev_end >= 0.5:
                    target_end = ((next_event.begin - prev_end) / 2.0) + prev_end
                    break

                prev_end = next_event.end

        # Make sure the target end doesn't land during an event
        assert (
            len(annotations[target_end]) == 0
        ), f"{target_end}: {annotations[target_end]}"

        # Slice out the fist segment_length of the audio sample
        seg_end = int(target_end * sr)
        x = audio[:seg_end]
        x_remain = audio[seg_end:]
        assert len(x) + len(x_remain) == len(audio)

        # Add the segmented audio to the splits, and recurse with remaining audio
        segmented_tree = IntervalTree(annotations[:target_end].copy())
        assert len(segmented_tree[target_end:]) == 0
        splits.append((x, segmented_tree))

        a_remaining = IntervalTree()
        for a in annotations[target_end:]:
            a_remaining.addi(a.begin - target_end, a.end - target_end, a.data)

        # Make sure that all the annotations are included in the saved annotations
        # remaining annotations
        assert len(segmented_tree) + len(a_remaining) == num_annotations
        splits = split_audio(x_remain, sr, a_remaining, previous_split=splits)
    else:
        splits.append((audio, annotations))

    return splits


def save_splits(
    pets: List[Tuple[Path, List[Path]]],
    file_splits: List,
    sample_rates: List,
    output_dir: Path,
):
    """
    Save the segmented audio files and annotations to disk
    """
    all_files = []
    for ((f, _), s, sr) in zip(pets, file_splits, sample_rates):
        name = f.stem.split("_")
        for i in range(len(s)):

            # Create filename with split index
            split_name = copy(name)
            split_name[0] = f"{name[0]}{i}"
            split_name = "_".join(split_name)
            split_name = f"{split_name}{f.suffix}"

            # Update the full file name with updated root
            filename = f.with_name(split_name)
            filename = str(filename).split("/")
            filename[0] = str(output_dir)
            filename = Path("/".join(filename))

            # Create the parent directories
            filename.parent.mkdir(parents=True, exist_ok=True)

            # Write segment audio file
            soundfile.write(filename, s[i][0], samplerate=sr)

            # Save file name
            all_files.append(filename)

            segment_annotations = {}
            for event in sorted(s[i][1]):
                if not event.data in segment_annotations:
                    segment_annotations[event.data] = []

                # Add separate lines for event on and off times
                segment_annotations[event.data].append(
                    (event.begin, f"on_{event.data}")
                )
                segment_annotations[event.data].append((event.end, f"off_{event.data}"))

            # Save annotations as a csv
            for key, item in segment_annotations.items():
                if filename.stem.split("_")[1] == "Iso":
                    annotation_fname = filename.with_name(
                        f"{Path(split_name).stem}.csv"
                    )
                else:
                    annotation_fname = filename.with_name(
                        f"{Path(split_name).stem}_tech_{key}.csv"
                    )
                df = pd.DataFrame(item)
                df.to_csv(annotation_fname, header=None, index=False)

    # Save list of all files -- sort before to ensure ordering is consistent
    log.info(f"Split dataset into {len(all_files)} segmented files")
    log.info(f"Saved segmented files and annotations to {output_dir}")
    log.info(f"Saving list of all files to file_names.txt")

    all_files = sorted(all_files)
    with open("file_names.txt", "w") as f:
        f.write("\n".join([str(f) for f in all_files]))


def segment_dataset(pets: list) -> Tuple[List[np.ndarray], List[float]]:
    """
    Segments audio files so all are less than 60 seconds.
    Applies the same segmentation to the annotations.
    """
    file_splits = []
    sample_rates = []
    for (f, a) in pets:
        x, sr = soundfile.read(f)
        sample_rates.append(sr)

        # Load annotations from csv into interval tree
        annos = load_annotations(a)

        # Split audio and annotations into < 60s segments
        splits = split_audio(x, sr, annos)
        file_splits.append(splits)

        # Confirm the duration and annotations make sense for segments
        for s in splits:
            # Segment duration
            duration = len(s[0]) / sr

            # Segment is shorter than 60 seconds
            assert duration < 60.0
            # There are no annotations beyond the segment duration
            assert len(s[1][duration:]) == 0, f"{s[1][duration:]}"

        # Make sure that the same number of segments exist
        assert sum([len(s[1]) for s in splits]) == len(annos)

        # Make sure that the joined audio is equal to the orignal audio
        joined = np.concatenate([s[0] for s in splits])
        assert np.all(x == joined)

    return file_splits, sample_rates


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-d", "--dataset", default="./CBFdataset", help="Path to CBF Dataset"
    )
    parser.add_argument(
        "-o",
        "--out",
        default="./CBFdataset_PETS",
        help="Output folder for PETs Dataset",
    )
    args = parser.parse_args(arguments)

    # Wav files and annotation files for PETs
    wavfiles = list(Path(args.dataset).rglob("*.wav"))
    log.info(f"Found {len(wavfiles)} wav files")

    annotation_files = get_annotation_files(wavfiles)
    log.info(f"{len(annotation_files)} files containing PETs")

    # Segment audio files and annotations
    file_splits, sample_rates = segment_dataset(annotation_files)

    # Save segmented audio files and annotations
    save_splits(annotation_files, file_splits, sample_rates, Path(args.out))


if __name__ == "__main__":
    log.info("Dataset Preprocessing")
    sys.exit(main(sys.argv[1:]))
