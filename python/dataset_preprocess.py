"""
Filters the CBF Dataset to only
include PETs (i.e. acciacatura, portamento, and glissando)
Segments audio files so all are less than 60 seconds.
"""
import logging
import os

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    log.info("Dataset Preprocessing")