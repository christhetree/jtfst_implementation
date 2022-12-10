"""
Filters the CBF Dataset to only
include PETs (i.e. acciacatura, portamento, and glissando)
Segments audio files so all are less than 60 seconds.
"""
import logging
import os
import sys
import argparse

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# PETs to include
tech_types = ["Glissando", "Portamento", "Acciacatura"]


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--in', default="./CBFdataset", help="Path to CBF Dataset")
    parser.add_argument('-o', '--out', default="./CBFdataset_PETS", help="Output folder for PETs Dataset")

    args = parser.parse_args(arguments)
    print(args)


if __name__ == "__main__":
    log.info("Dataset Preprocessing")
    sys.exit(main(sys.argv[1:]))
