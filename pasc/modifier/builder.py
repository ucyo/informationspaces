#!/usr/bin/env python
# coding: utf-8
"""Build modifier for Infospace extraction of Sequence."""

from itertools import chain
from collections import defaultdict
# from pasc.backend import BaseBuilder
from pasc.objects.sequence import Sequence  # Input
from pasc.objects.informationspace import InformationSpace  # Output
from pasc.objects.informationcontext import InformationContext as IC
from pasc.toolbox import flood as fl
import numpy as np
#
#
# class Builder(BaseBuilder):
#
#     name = "Builder"
#
#     @staticmethod
#     def build_informationspace(seq):
#         assert isinstance(seq, Sequence)
#         if seq.sequence.size == 0:
#             return None
#         arr = np.ones(shape=seq.shape, dtype=seq.dtype) * fl.getNAN(seq.dtype)
#         for i, idx in enumerate(seq.sequence):
#             arr.flat[idx] = seq.data[i]
#
#         # TODO: Fix this by using the idx instead of actual value
#         # TODO: Also in restricted builder
#         fillvalue = np.max(seq.data) + 1
#         arr.flat[seq.sequence[-1]] = fillvalue
#
#         space = _buildInfoSpace(arr=arr, val=fillvalue)
#         result = dict()
#         for dim, arrs in space.items():
#             tmp = list()
#             for arr in arrs:
#                 arr[np.where(arr == fillvalue)] = fl.getNAN(seq.dtype)
#                 tmp.append(arr)
#             result[dim] = tmp
#         return InformationSpace(result)
#
#     def build_infospace(self, array, fillvalue, nanvalue):
#         # array = array.copy()
#         space = _buildInfoSpace(arr=array, val=fillvalue)
#         result = dict()
#         for dim, arrs in space.items():
#             tmp = list()
#             for arr in arrs:
#                 arr[np.where(arr == fillvalue)] = nanvalue
#                 tmp.append(arr)
#             result[dim] = tmp
#         return InformationSpace(result)
#
#
# class RestrictedBuilder(BaseBuilder):
#
#     name = "RestrictedBuilder"
#
#     def __init__(self, restriction):
#         self.restriction = restriction
#
#     def _get_restriction(self):
#         return self._restriction
#
#     def _set_restriction(self, value):
#         if not isinstance(value, int):
#             err = "Restriction must be an integer"
#             raise TypeError(err)
#         self._restriction = value
#     restriction = property(_get_restriction, _set_restriction)
#
#     def build_informationspace(self, seq):
#         assert isinstance(seq, Sequence)
#         if seq.sequence.size == 0:
#             return None
#
#         arr = np.ones(shape=seq.shape, dtype=seq.dtype) * fl.getNAN(seq.dtype)
#         for i, idx in enumerate(seq.sequence):
#             arr.flat[idx] = seq.data[i]
#
#         fillvalue = np.max(seq.data) + 1
#         arr.flat[seq.sequence[-1]] = fillvalue
#
#         arr = _getSubssetInRangeOf(arr, self.restriction, fillvalue)
#         space = _buildInfoSpace(arr=arr, val=fillvalue)
#         result = dict()
#         for dim, arrs in space.items():
#             tmp = list()
#             for arr in arrs:
#                 arr[np.where(arr == fillvalue)] = fl.getNAN(seq.dtype)
#                 tmp.append(arr)
#             result[dim] = tmp
#         return InformationSpace(result)
#
#     def build_infospace(self, array, fillvalue, nanvalue):
#         # array = array.copy()
#         array = _getSubssetInRangeOf(array, self.restriction, fillvalue)
#         space = _buildInfoSpace(arr=array, val=fillvalue)
#         result = dict()
#         for dim, arrs in space.items():
#             tmp = list()
#             for arr in arrs:
#                 arr[np.where(arr == fillvalue)] = nanvalue
#                 tmp.append(arr)
#             result[dim] = tmp
#         return InformationSpace(result)


class GeneralBuild:

    @staticmethod
    def build_infospace(array, searchval=None, searchidx=None, restriction=None):
        intnan = fl.getNAN(array.dtype)
        if searchval:
            origin = np.argwhere(array == searchval)[0]
        elif searchidx:
            origin = np.unravel_index(searchidx, array.shape)
        else:
            raise Exception("No value or index given.")
        sval = array[tuple(origin)]
        if restriction is not None:
            array = _getSubssetInRangeOf(array, restriction, sval)
        spacedict = _buildInfoSpace(arr=array, val=sval)
        space = _set_searchval_to_intnan(spacedict, sval, intnan)
        return space

#
# class OneDBuild:
#
#     @staticmethod
#     def build_infospace(array, searchval=None, searchidx=None, restriction=None):
#         intnan = fl.getNAN(array.dtype)
#         if searchval:
#             origin = np.argwhere(array == searchval)[0]
#         elif searchidx:
#             origin = np.unravel_index(searchidx, array.shape)
#         else:
#             raise Exception("No value or index given.")
#         sval = array[origin]
#
#         # Candidates are slices going through searchval
#         candidates = []
#         for i in range(len(origin)):
#             tmp = list(origin)
#             tmp[i] = slice(None, None, None)
#             candidates.append(tmp)
#
#         # Result is 1d array going forward (+1)
#         # and backward (-1) in each dimension
#         result = []
#         for i, x in enumerate(candidates):
#             source = origin[i]
#             assert array[x][source] == sval, "Error in indexing: {} != {}".format(
#                 array[x][source], sval)
#             dim = []
#             for direction in [-1, +1]:
#                 direc = 0
#                 while ((source + direc) >= 0 and
#                        (source + direc) < array[x].size and
#                        array[x][source + direc] != intnan):
#                     if restriction and abs(direc) > restriction:
#                         break
#                     direc += direction
#                 v = source + direc + -direction
#                 dim.append(v)
#             if min(dim) == max(dim):
#                 continue
#             elif (min(dim) == source - 1 and
#                   max(dim) == source + 1):
#                 continue
#             # else slice(source,v+1)
#             sli = slice(min(dim), max(dim) + 1, None)
#             tmp = array[x].copy()
#             tmp[source] = intnan
#             result.append(tmp[sli].copy())
#         ispace = InformationSpace({1: IC(result)})
#         return ispace


def _set_searchval_to_intnan(ispace, searchval, intnan, **info):
    result = dict()
    for dim, arrs in ispace.items():
        tmp = list()
        for arr in arrs:
            j = arr.copy()
            j[np.where(j == searchval)] = intnan
            tmp.append(IC.create(data=j, **info))
        result[dim] = tmp
    return InformationSpace(result)

#
# class OneDBuilder(RestrictedBuilder):
#
#     def build_infospace(self, data, searchval, nanvalue):
#         data = data.copy()
#         origin = np.argwhere(data == searchval)[0]
#         INTNAN = fl.getNAN(data.dtype)
#
#         # Candidates are slices going through searchval
#         candidates = []
#         for i in range(len(origin)):
#             tmp = list(origin)
#             tmp[i] = slice(None, None, None)
#             candidates.append(tmp)
#
#         # Result is 1d data going forward (+1)
#         # and backward (-1) in each dimension
#         result = []
#         for x in candidates:
#             # if (data[x] != INTNAN).all():
#             #     result.append(data[x])
#             #     continue
#             source = np.argwhere(data[x] == searchval)[0][0]
#             dim = []
#             for direction in [-1, +1]:
#                 direc = 0
#                 while (source + direc) >= 0 and (source + direc) < data[x].size and data[x][source + direc] != INTNAN:
#                     if self.restriction and abs(direc) > self.restriction:
#                         break
#                     direc += direction
#                 v = source + direc + -direction
#                 dim.append(v)
#             if min(dim) == max(dim):
#                 continue
#             sli = slice(min(dim), max(dim) + 1, None)  # else slice(source,v+1)
#             result.append(data[x][sli].copy())
#
#         # Set searchval to INTNAN for further processing
#         for arr in result:
#             arr[np.argwhere(arr == searchval)] = nanvalue
#         # Ispace is final information space
#         ispace = InformationSpace({1: IC(result)})
#         return ispace


def _getSubssetInRangeOf(data, r, value):
    coords = np.array([x[0] for x in np.where(data == value)])
    lower = np.array([max(a, b)
                      for a, b in zip([0] * coords.size, coords - r)])
    upper = np.array([min(a, b) for a, b in zip(data.shape, coords + r)])
    subset = data[[slice(x, y + 1, None) for x, y in zip(lower, upper)]]
    return subset  # .copy()


def _buildInfoSpace(arr, val, squeeze=True):
    """Build Infospace"""
    blocks = fl.getBlocks(arr, val)  # blocks of different dimensions
    test = fl.ndimDict(blocks)  # dictionary of blocks with ndim as keys
    chained = chain(*[v for k, v in test.items()])  # reduction of nested lists
    reduced = [fl.reduceTill(ar, val, squeeze=squeeze) for ar in chained]
    dicts = [fl.ndimDict(x, uniques=True, squeeze=squeeze) for x in reduced]

    dict3 = defaultdict(list)
    for k, v in chain(*[x.items() for x in dicts]):
        dict3[k].extend(v)
    infospace = {k: fl.uniqueNDsAndSuper(v) for k, v in dict3.items()}
    try:
        del infospace[0]
    except KeyError:
        pass
    infospace[0] = [np.array([val], dtype=int)]
    empty = [x for x in range(arr.ndim + 1) if x not in infospace.keys()]
    for i in empty:
        infospace[i] = np.array([], dtype=int)
    return infospace

#
# class OneDBuildMA:
#
#     @staticmethod
#     def build_infospace(array, searchval=None, searchidx=None, restriction=None):
#         intnan = fl.getNAN(array.dtype)
#         if searchval is not None:
#             origin = np.argwhere(array == searchval)[0]
#         elif searchidx is not None:
#             origin = np.unravel_index(searchidx, array.shape)
#         else:
#             raise Exception(
#                 "No value or index given, got {} {}".format(searchval, searchidx))
#         sval = array[origin]
#
#         # Candidates are slices going through searchval
#         candidates = []
#         for i in range(len(origin)):
#             tmp = list(origin)
#             tmp[i] = slice(None, None, None)
#             candidates.append(tmp)
#
#         # Result is 1d array going forward (+1)
#         # and backward (-1) in each dimension
#         result = []
#         for i, x in enumerate(candidates):
#             source = origin[i]
#             assert array[x][source] == sval, "Error in indexing: {} != {}".format(
#                 array[x][source], sval)
#             dim = []
#             for direction in [-1, +1]:
#                 direc = 0
#                 while ((source + direc) >= 0 and
#                        (source + direc) < array[x].size and
#                        array.mask[x][source + direc] == False):
#                     if restriction and abs(direc) > restriction:
#                         break
#                     direc += direction
#                 v = source + direc + -direction
#                 dim.append(v)
#             if min(dim) == max(dim):
#                 continue
#             elif (min(dim) == source - 1 and
#                   max(dim) == source + 1):
#                 continue
#             # else slice(source,v+1)
#             sli = slice(min(dim), max(dim) + 1, None)
#             tmp = array.data[x].copy()
#             tmp[source] = intnan
#             result.append(tmp[sli].copy())
#         ispace = InformationSpace({1: IC(result)})
#         return ispace
#


class OneDBuildMA:

    @staticmethod
    def build_infospace(array, searchval=None, searchidx=None, restriction=None):
        intnan = fl.getNAN(array.dtype)
        if searchval is not None:
            origin = np.argwhere(array == searchval)[0]
        elif searchidx is not None:
            origin = np.unravel_index(searchidx, array.shape)
        else:
            raise Exception(
                "No value or index given, got {} {}".format(searchval, searchidx))
        sval = array[origin]

        # Candidates are slices going through searchval
        candidates = []
        for i in range(len(origin)):
            tmp = list(origin)
            tmp[i] = slice(None, None, None)
            candidates.append(tmp)

        # Result is 1d array going forward (+1)
        # and backward (-1) in each dimension
        result = []
        for i, x in enumerate(candidates):
            source = origin[i]
            assert array[x][source] == sval, "Error in indexing: {} != {}".format(
                array[x][source], sval)
            tmp = array[x].copy()
            tmp.data[source] = intnan
            if restriction is not None:
                tmp = tmp[slice(max(source - restriction, 0),
                                min(source + restriction, array[x].size) + 1)]
            if tmp.data[tmp.mask == False].size > 1:
                data = tmp.data[tmp.mask == False]
                result.append(IC.create(data=data, id=(i,), size=data.size))
        ispace = InformationSpace({1: IC(result)})
        return ispace
