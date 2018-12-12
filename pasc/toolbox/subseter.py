#!/usr/bin/env python
# coding: utf-8
"""
Subsetting FloatArray's for faster comparison.
"""

from itertools import combinations, product
import numpy as np
from pasc.toolbox.synthetics import logseed
from pasc.objects.floatarray import FloatArray
import xarray as xr


class Subseter:

    name = "Subseter"

    @staticmethod
    @logseed
    def subset(obj, size, error=.005):
        if isinstance(obj, np.ndarray):
            subs = SubsetSlices(obj.shape, size, error)
            result = obj[subs]
        elif isinstance(obj, xr.DataArray):
            subs = SubsetSlices(obj.values.shape, size, error)
            slices = {x: y for x, y in zip(obj.dims, subs)}
            result = obj.isel(**slices)
        elif isinstance(obj, FloatArray):
            subs = SubsetSlices(obj.array.shape, size, error)
            result = FloatArray(obj.array[subs])
        else:
            raise TypeError("Can't understand type.")
        return result


def pairs(*lists):
    ndim = len(lists)
    for t in combinations(lists, ndim):
        for pair in product(*t):
            yield pair


def getPairs(shape, N, err):
    paris = pairs(*(list(range(dim + 1)) for dim in shape))
    for pair in paris:
        size = np.prod(pair)
        if size >= N * (1 - err):
            if size <= N * (1 + err):
                yield pair


def getRandomSubsetN(shape, N, err):
    pairs = getPairs(shape, N, err)
    ally = list(pairs)
    np.random.shuffle(ally)
    return ally[0]


def SubsetSlices(shape, N, err):
    subsetSizes = getRandomSubsetN(shape, N, err)
    startidx = tuple(np.random.randint(0, x - y + 1)
                     for x, y in zip(shape, subsetSizes))
    result = [slice(x, x + y, None) for x, y in zip(startidx, subsetSizes)]
    return result
