#!/usr/bin/env python
# coding=utf-8
"""One hit wonder functions for different tasks."""

import pkg_resources
from xarray import open_dataset
from pyevtk.hl import gridToVTK
from pasc.toolbox.flood import getNAN
import numpy as np


_datamapping = {'pre': pkg_resources.resource_filename(
    'pasc', 'data/sresa1b_ncar_ccsm3-example.nc')}


def setrandomNANs(arr, size=None, seed=None):
    """Set random NAN values to IndexArray."""
    INTNAN = getNAN(arr.dtype)

    def _randomNAN(mi, ma, size):
        return np.random.randint(mi, ma, size)

    if seed:
        np.random.seed(seed)
    mi = 1
    ma = arr.size
    if not size:
        size = _randomNAN(mi, ma, 1)
    result = arr.copy()

    for i in _randomNAN(mi, ma, size):
        result.flat[i] = INTNAN
    return result


def generateIndexArray(shape):
    """Generate IndexArray by using a shape."""
    return np.arange(np.prod(shape)).reshape(shape)


def generateRandomIndexArray(shape, size=None, seed=None):
    """Generate random IndexArray by using a shape."""
    tmp = generateIndexArray(shape=shape)
    result = setrandomNANs(arr=tmp, size=size, seed=seed)
    return result


def check_methods(C, *methods):
    """Check if C implemented *methods.

    INFO
    ====
    Source: python3.6/_collections_abc.py
    """
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


def load_data(key, **kwargs):
    """Load file from example data."""
    path = get_data_path(key=key)
    if not path:
        raise KeyError('Unknown datafile.')
    return open_dataset(path, **kwargs)


def get_data_path(key):
    """Get example data paths."""
    return _datamapping.get(key, False)


def np2vtr(arr, arrname, output):
    """Save numpy array as vtr file."""
    assert arr.ndim == 3, "3 Dim needed."
    arr = arr.astype(float)
    xs, ys, zs = arr.shape
    x, y = np.arange(0, xs + 1), np.arange(0, ys + 1)
    z = np.arange(0, zs + 1)
    fname = gridToVTK(output, x, y, z, cellData={arrname: arr})
    return fname

def get_bits(data):
    if data.dtype in (np.int64, np.uint64):
        bits = 64
    elif data.dtype in (np.int32, np.uint32):
        bits = 32
    else:
        err = "Expedted (u)int of 32 or 64 Bit, got {}".format(data.dtype)
        raise TypeError(err)
    return bits
