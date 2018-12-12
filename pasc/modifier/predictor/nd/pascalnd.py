#!/usr/bin/env python
# coding=utf-8
"""
Method for getting Pascal weights for N-Dimensional setup and arbitary position
of x.

NOTE
====
All dimensions have the same weights!
"""
import logging
from pasc.toolbox.flood import getNAN
import numpy as np
from scipy import linalg
_log = logging.getLogger(__name__)


def pascalnd(shape, xpos):
    """Get n dimensional Pascal weights. Arbitary position of searched value.

    Note
    ====
    All dimensions have the same weights.

    Arguments
    =========
    shape : tuple
        Tuple defining the shape of the weights.
    xpos : tuple
        Tuple defining the index position of np.nan value representing
        the point in the shape which should be interpolated.

    Result
    ======
    result : np.ndarray
        Numpy array with pascal weights.
    """
    if not isiterable(shape):
        shape = (shape,)
    if not isiterable(xpos):
        xpos = (xpos,)
    assert len(shape) == len(xpos), "Length of inputs does not match."
    arrs = [_getarrx(x, y) for x, y in zip(shape, xpos)]

    prev = arrs[0]
    result = prev
    for i, arr in enumerate(arrs):
        if i == 0:
            continue
        curr = np.tensordot(arr, prev, 0) * -1
        replace = xpos[:i][::-1]
        curr[xpos[i], :] = prev
        ind = [slice(None)] + list(replace)
        curr[ind] = arr
        prev = curr
        result = curr.T
    return result


def isiterable(obj):
    """Check if obj is iterable."""
    return hasattr(obj, "__iter__")


def estimate(data):
    """Get estimate for np.nan value with pascal weights.

    Arguments
    =========
    data : np.ndarray (with one np.nan value)
        An array with np.nan at the position which should be estimated.

    Result
    ======
    result : numeric
        Estimated value for position with np.nan
    """
    xpos = _get_nan_position(data)
    weights = pascalnd(data.shape, xpos)
    result = np.nansum(data * weights)
    return result


def fill(data):
    """Get new data w/o np.nan value but the pascal estimate."""
    data = data.copy()
    xpos = _get_nan_position(data)
    val = estimate(data)
    data[xpos] = val
    return data


def test_balance(arr):
    """
    Check if the weights are balanced out. This means that every dimensions
    is same weighted and gets a value of 1.
    """
    # TODO: Might add the position of 1. Which should be exactly where the nan value is in the array
    for i in range(arr.ndim):
        sums = np.nansum(arr, axis=i)
        yield np.sum(sums) == 1


def _get_pascal_weights(depth):
    """
    Get Pascal pyramid weights for arbitary depth/size.
    """
    assert 0 < depth < 10, "Depth must be between [0,10], got {}".format(depth)
    depth = depth + 1
    # TODO: expm3 is deprecated, but expm can;t work with depth 7 and 8
    # Issue 8029: https://github.com/scipy/scipy/issues/8029
    diag = linalg.expm3(np.diag(np.arange(1, depth), -1))
    signed = diag[1:, 1:].dot(np.diag(
        [x if i % 2 == 0 else -x for i, x in enumerate([1] * (depth - 1))])).astype(int)
    if depth > 2:
        signed[1, :] = np.array([-1, 2] + [0] * (depth - 3))  # fix first row
    if depth > 1:
        signed[0, :] = np.array([1] + [0] * (depth - 2))  # fix second row
    return signed if depth > 1 else signed[0]


def _pascal1d(length):
    """Get Pascal weights with length x."""
    arr = _get_pascal_weights(length)
    return arr[-1][::-1] if length > 2 else arr[-1]


def _getarrx(N, x):
    """Get one dimensional weights."""
    arr = _pascal1d(N - 1)  # pascal[N-1]
    assert 0 <= x <= len(arr), "X is not in valid range"
    corner = x == len(arr) or x == 0
    if not corner:
        tmp = arr / abs(arr[x])
        tmp = tmp * -1 if arr[x] > 0 else tmp
        add = 1. / abs(arr[x]) if arr[x] > 0 else -1. / abs(arr[x])
        tmp = np.append(tmp, add)
        tmp[x] = np.nan
    else:
        tmp = np.append(arr, np.nan)
    if x == 0:
        tmp = tmp[::-1]
    return tmp


def _get_nan_position(data):
    """Find index position of np.nan value.

    Note
    ====
    Now supports all kind of dtypes.
    """
    if data.dtype in (np.float32, np.float64):
        nans = tuple(np.transpose(np.isnan(data).nonzero()))
    else:
        nanval = getNAN(data.dtype)
        nans = tuple(np.transpose((data == nanval).nonzero()))
    if not nans:
        raise Exception("Non np.nan value found.")
    if len(nans) > 1:
        message = "Only one np.nan value allowed, found {}."
        raise Exception(message.format(len(nans)))
    return nans[0]
