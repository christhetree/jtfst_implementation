import logging
import os
from typing import Optional

import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_scalogram(x: T, wavelet: Optional[T] = None) -> T:
    pass


if __name__ == "__main__":
    log.info("Calculating scalogram")
    batch_size = 1
    n_ch = 2
    n_samples = 48000
    audio = tr.rand((batch_size, n_ch, n_samples))
    scalogram = calc_scalogram(audio)
