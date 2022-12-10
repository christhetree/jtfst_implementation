import argparse
import logging
import os
from pathlib import Path
import sys

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", help="Input file to be processed", type=str,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="./features",
        help="Output folder for preprocessed features",
    )

    args = parser.parse_args(arguments)
    print(args)


if __name__ == "__main__":
    log.info("Feature Preprocessing")
    sys.exit(main(sys.argv[1:]))