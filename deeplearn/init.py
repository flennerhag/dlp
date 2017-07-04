"""
Initialization routines.
"""

import numpy as np


def init_weights(fan_in, fan_out, scale=1., resize=True, den=2., dist=None):
    """
    Random initialization of weight matrix.

    Args:
        fan_in (int): ingoing dimensionality (n_rows)
        fan_out (int): ingoing dimensionality (n_cols)
        scale (int): scale weights
        resize (bool): whether to use inverse square rescaling
        den (float, int): denominator in resizing
        dist (obj): numpy distribution to use.
    """
    if dist is None:
        dist = np.random.randn

    W = scale * dist(fan_in, fan_out)
    if resize:
        W /= np.sqrt(fan_in / den)

    return W.astype(np.float32)


def init_filter(fan_in, fan_out, depth,
                scale=1., resize=True, den=2., dist=None):
    """
    Random initialization of convolution filters .

    Args:
        fan_in (int): ingoing dimensionality (n_rows)
        fan_out (int): ingoing dimensionality (n_cols)
        scale (int): scale weights
        resize (bool): whether to use inverse square rescaling
        den (float, int): denominator in resizing
        dist (obj): numpy distribution to use.
    """
    if dist is None:
        dist = np.random.randn

    W = scale * dist(fan_in, fan_out, depth)
    if resize:
        W /= np.sqrt(fan_in / den)

    return W.astype(np.float32)


def init_bias(fan_out, scale=0., dist=None):
    """
    Random initialization of bias vector.

    Args:
        fan_out (int): dimensionality (n_cols)
        dist (obj, None): distribution for initialization
        scale (float): rescaling
    """
    if dist is None:
        dist = np.random.randn

    b = scale * dist(fan_out).reshape(1, fan_out)

    return b.astype(np.float32)
