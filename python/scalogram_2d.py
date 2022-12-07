import logging
import os
from typing import Optional, List

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from tqdm import tqdm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_scalogram_2d_td(scalogram: T, wavelet_bank: List[T], take_modulus: bool = True) -> T:
    assert scalogram.ndim == 3
    assert wavelet_bank

    scalogram = scalogram.unsqueeze(1)  # Image with 1 channel
    scalogram_complex = scalogram.to(wavelet_bank[0].dtype)
    convs = []
    for wavelet in tqdm(wavelet_bank):
        assert wavelet.ndim == 2
        kernel = wavelet.view(1, 1, *wavelet.shape)
        out = F.conv2d(scalogram_complex, kernel, stride=(1, 1), padding="same")
        convs.append(out)

    jtfst = tr.cat(convs, dim=1)
    if take_modulus:
        jtfst = tr.abs(jtfst)

    return jtfst


def calc_scalogram_2d(x: T,
                      wavelet_bank: List[T],
                      max_f_dim: Optional[int] = None,
                      max_t_dim: Optional[int] = None,
                      take_modulus: bool = True) -> T:
    assert x.ndim == 3
    if max_f_dim is None:
        max_f_dim = max([w.size(0) for w in wavelet_bank])
    if max_t_dim is None:
        max_t_dim = max([w.size(1) for w in wavelet_bank])
    max_f_padding = max_f_dim // 2
    max_t_padding = max_t_dim // 2
    # TODO(cm): check why we can get away with only padding the front
    x = F.pad(x, (max_t_padding, 0, max_f_padding, 0))
    log.debug("scalogram fft")
    x_fd = tr.fft.fft2(x).unsqueeze(1)

    log.debug("making kernels")
    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 2
        top_padding = max_f_padding - wavelet.size(-2) // 2
        bottom_padding = x_fd.size(-2) - wavelet.size(-2) - top_padding
        left_padding = max_t_padding - wavelet.size(-1) // 2
        right_padding = x_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, *wavelet.shape)
        kernel = F.pad(kernel, (left_padding, right_padding, top_padding, bottom_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=1)
    log.debug("kernel fft")
    kernels_fd = tr.fft.fft2(kernels)
    log.debug("matmult")
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    y_fd = kernels_fd * x_fd
    log.debug("y ifft")
    y = tr.fft.ifft2(y_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    y = y[:, :, :-max_f_padding, :-max_t_padding]
    if take_modulus:
        y = tr.abs(y)
    return y
