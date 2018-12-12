#!/usr/bin/env python
# coding: utf-8
"""Algorithm for building of InfoSpace inspired by floodfill algorithms."""

from functools import lru_cache
from collections import deque  # , defaultdict
from itertools import groupby, chain
import numpy as np


@lru_cache(32)
def getNAN(dtype):
    if not isinstance(dtype, np.dtype):
        err = "Expected np.dtype, got {}".format(type(dtype))
        raise TypeError(err)
    return np.array([np.nan]).astype(dtype)[0]


class NoneT:
    """Special None type for concious setting to None."""

    def __repr__(self):
        return "All"


ALL = NoneT()


def hasNAN(arr):
    """There exists a NaN value in IntegerArray."""
    INTNAN = getNAN(arr.dtype)
    return np.any(arr == INTNAN)


def getAllNANcoords(arr):
    """All NaN values in IntegerArray."""
    INTNAN = getNAN(arr.dtype)
    return zip(*(arr == INTNAN).nonzero())


def slicedice(tupl):
    """Generate slices along each element in tuple of corrosponding dimension."""
    size = len(tupl)
    alls = [ALL] * size

    slices = list()
    for i, t in enumerate(tupl):
        tmp = alls.copy()
        tmp[i] = t
        slices.append(tuple(tmp))
    return slices


def slicecut(arr, valcoord, tupl):
    """Actualslicing of array.

    Actual slicing of array at position tupl and continue with subarrays
    containing val.
    """
    default = slice(None, None, None)
    slicetuple = list()
    for i, x in enumerate(tupl):
        if not isinstance(x, NoneT):
            if valcoord[i] == x:
                return None
            elif valcoord[i] > x:
                slicetuple.append(slice(x + 1, None, None))
            else:
                slicetuple.append(slice(None, x, None))
        else:
            slicetuple.append(default)
    result = arr.copy()
    return result[slicetuple]


def getBlocks(arr, val):
    """Identify blocks of valid entries.

    Main algorithm applied to array for identifiying
    neighbouring blocks containing val.
    """
    todo = deque()
    done = deque()
    arr = fence(val, arr)
    todo.append(arr)

    while todo:
        arr = todo.pop()
        if arr is None:
            continue
        try:
            valcoord = list(zip(*np.where(arr == val)))[0]
        except IndexError:
            continue
        nanpos = list(getAllNANcoords(arr))
        if nanpos:
            slices = [slicedice(x) for x in nanpos]
            # flat = [item for sublist in slices for item in sublist]
            flat = list(chain.from_iterable(slices))
            for sli in set(flat):
                # choose best sli
                obj = slicecut(arr, valcoord, sli)
                if obj is not None:
                    todo.append(obj)
        elif not all([x is None for x in arr]):
            # arr.flags.writeable = False
            done.append(arr)
    return done


# TODO: Can be improved if a search limit is being used. No predictor will use a maximum k elements in a single direction. This can be restricted here.
def fence(searchval, data):
    """Dramatically sink search space for adjacent blocks.

    Rooted at the searchval one traverses in each dimension
    until one arrives the dimension limits or a INTNAN value.
    These are the defining borders for the search space of
    consecutive blocks.
    """
    origin = tuple([x[0] for x in np.where(data == searchval)])
    coords = list(origin)
    INTNAN = getNAN(data.dtype)

    corners = []
    for dim in range(data.ndim):
        pack = []
        for direction in [-1, +1]:
            current = data[origin]
            coords = list(origin)
            while current != INTNAN:
                coords[dim] += direction
                if any([x < 0 for x in coords]):
                    coords[dim] -= direction
                    break
                try:
                    current = data[tuple(coords)]
                except IndexError:
                    break
            pack.append(coords[dim])
        corners.append(pack)
    newdata = data[[slice(x, y, None) if x < y else slice(
        y, x, None) for x, y in corners]]
    return newdata


def getUniqueBlocks(arr, val):
    """Get unique blocks of `array` with value `val`."""
    blocks = getBlocks(arr=arr, val=val)
    result = uniqueNDs(blocks)
    return result


class _wrapper(object):
    """Helper class to make np.ndarray hashable."""

    def __init__(self, array):
        self._array = array
        self._hash = hash(array.data.tobytes())

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # and np.array_equal(self._array, other._array)
        return self._hash == other._hash


def uniqueNDs(arrs):
    """Identify unique np.ndarrays from a list of values."""
    return list(u._array for u in set([_wrapper(x) for x in arrs]))


def uniqueNDsAndSuper(arrs):
    """Remove subsets of collection of arrays"""
    uniq = uniqueNDs(arrs)
    result = get_supers(uniq)
    return result


def reduce1dim(arr, val):
    """Reduce array at value to ndim subarrays."""
    if arr.squeeze().ndim == 0:
        return [arr]
    coord = np.where(arr == val)
    pos = list(zip(*coord))[0]
    slices = slicedice(pos)
    slicetuples = list()
    for ones in slices:
        tmp = list()
        for x in ones:
            if not isinstance(x, NoneT):
                tmp.append(slice(x, x + 1, None))
            else:
                tmp.append(slice(None, None, None))
        slicetuples.append(tmp)
    return [arr[x] for x in slicetuples]


# TODO: Squeeze not working. It runs endless. Reproduce by
# TODO: shape = (3,5,3); val=36; nans=1; seed=42354; arr = tb.generateRandomIndexArray(shape, nans, seed=seed)
def reduceTill(arr, val, ndim=None, squeeze=True):
    """Reduce ND array till `ndim` dimensions."""
    squeeze = True
    todo = deque()
    done = deque()
    todo.append(arr)

    while todo:
        arr = todo.pop()
        if ndim:
            if arr.ndim <= ndim:
                done.append(arr)
                break
            subs = reduce1dim(arr, val)
            for a in subs:
                if a.squeeze().ndim == ndim:
                    aq = a.squeeze() if squeeze else a
                    done.append(aq)
                else:
                    aq = a.squeeze() if squeeze else a
                    todo.append(aq)
        else:
            aq = arr.squeeze() if squeeze else arr
            done.append(arr)
            subs = reduce1dim(arr, val)
            for a in subs:
                if a.size != 1:
                    aq = a.squeeze() if squeeze else a
                    done.append(aq)
                    todo.append(aq)
    return done


def ndimDict(arrs, uniques=False, squeeze=False):
    """Turn list of ndarrays to dict of ndarrays with ndim as key."""
    if uniques:
        arrs = uniqueNDs(arrs=arrs)
    if squeeze:
        arrs = [a.squeeze() for a in arrs]
    result = dict()
    sorted_done = sorted(arrs, key=lambda x: x.ndim)
    for k, v in groupby(sorted_done, lambda x: x.ndim):
        result[k] = list(v)
    return result


def get_supers(arrs):
    """Remove all subsets from a list of arrays."""
    sorted_li = sorted(arrs, key=lambda x: x.size)
    supers = list()
    while sorted_li:
        arr = sorted_li.pop(0)
        include = True
        for rest in sorted_li:
            if all([x in rest for x in arr.flat]):
                include = False
                break
            # print(arr, rest, all([x in rest.flat for x in arr.flat]))
        if include:
            supers.append(arr)
    return supers
