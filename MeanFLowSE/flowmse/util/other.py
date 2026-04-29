"""Utility helpers used by the flowmse package.

These were originally located in the top-level utils.py; collected here so
that `flowmse.util.other` can be imported as expected by model.py and
inference.py.
"""

import numpy as np
import torch


def si_sdr(s, s_hat):
    """Scale-invariant SDR (numpy, single example)."""
    alpha = np.dot(s_hat, s) / np.linalg.norm(s) ** 2
    sdr = 10 * np.log10(
        np.linalg.norm(alpha * s) ** 2 / np.linalg.norm(alpha * s - s_hat) ** 2
    )
    return sdr


def pad_spec(Y):
    """Right-pad a (B, C, F, T) spectrogram so that T is a multiple of 64."""
    T = Y.size(3)
    if T % 64 != 0:
        num_pad = 64 - T % 64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0, 0))
    return pad2d(Y)
