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
    non_pets = []
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

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dataset', default="./CBFdataset", help="Path to CBF Dataset")
    parser.add_argument('-o', '--out', default="./CBFdataset_PETS", help="Output folder for PETs Dataset")

    args = parser.parse_args(arguments)

    # Wav files and annotation files for PETs
    wavfiles = list(Path(args.dataset[0]).rglob("*.wav"))
    annotation_files = get_annotation_files(wavfiles)

    print(annotation_files)


if __name__ == "__main__":
    log.info("Dataset Preprocessing")
    sys.exit(main(sys.argv[1:]))
