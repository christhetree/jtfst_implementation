"""
Filters the CBF Dataset to only
include PETs (i.e. acciacatura, portamento, and glissando)
Segments audio files so all are less than 60 seconds.
"""
import logging
import os
import sys
import argparse
from pathlib import Path

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import soundfile

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# PETs to include
tech_types = ["Glissando", "Portamento", "Acciacatura"]

def get_annotation_files(wavfiles: list):
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

def load_annotations(annotations):
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
            begin = times[2*k]
            end = times[2*k + 1]
            tree.addi(begin, end, tech)
    return tree

def split_audio(audio, sr, annotations, segment_length=30.0, previous_split=None):
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
        if len(annotations[target_end-0.5:target_end+0.5]):
            a_iter = iter(sorted(annotations[target_end:]))
            prev_end = next(a_iter).end
            while True:
                next_event = next(a_iter)
                if next_event.begin - prev_end >= 0.5:
                    target_end = ((next_event.begin - prev_end) / 2.0) + prev_end
                    break
                
                prev_end = next_event.end
        
        # Make sure the target end doesn't land during an event
        assert len(annotations[target_end]) == 0, f"{target_end}: {annotations[target_end]}"

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

def segment_dataset(pets: list):
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

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dataset', default="./CBFdataset", help="Path to CBF Dataset")
    parser.add_argument('-o', '--out', default="./CBFdataset_PETS", help="Output folder for PETs Dataset")
    args = parser.parse_args(arguments)

    # Wav files and annotation files for PETs
    wavfiles = list(Path(args.dataset).rglob("*.wav"))
    log.info(f"Found {len(wavfiles)} wav files")

    annotation_files = get_annotation_files(wavfiles)
    log.info(f"{len(annotation_files)} files containing PETs")


if __name__ == "__main__":
    log.info("Dataset Preprocessing")
    sys.exit(main(sys.argv[1:]))
