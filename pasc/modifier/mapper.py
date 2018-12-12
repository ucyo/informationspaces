#!/usr/bin/env python
# coding: utf-8
"""Mapper modifier to transform FloatArray to IntegerArray."""

import struct
from pasc.backend import BaseMapper
from pasc.objects.floatarray import FloatArray  # Input
from pasc.objects.integerarray import IntegerArray  # Output
import numpy as np
from bitstring import BitArray as ba


class RawBinary(BaseMapper):
    """Raw binary mapper. Reads float binary and interprets it as integer."""

    name = "RawBinary"

    @staticmethod
    def map(floatarray):
        if not isinstance(floatarray, FloatArray):
            err_type = "Expected FloatArray, got {}".format(type(floatarray))
            raise TypeError(err_type)
        if floatarray.array.dtype in (np.float32,):
            i, o, d = ('>f', '>l', np.int32)
        elif floatarray.array.dtype in (np.float64, float,):
            i, o, d = ('>d', '>Q', np.int64)
        else:
            err_msg = 'Expected 32 or 64 bits, got {}'.format(floatarray.array.dtype)
            raise TypeError(err_msg)
        data = _vraw(floatarray.array, i, o).astype(d)
        return IntegerArray(data)

def _raw(value, itype, otype):
    s = struct.pack(itype, value)
    return struct.unpack(otype, s)[0]
_vraw = np.vectorize(_raw)


class Lindstrom(BaseMapper):
    """Map float to uint with saving order (based on Lindstrom et al. 2004)."""

    name = "Lindstrom"

    @staticmethod
    def map(floatarray):
        if not isinstance(floatarray, FloatArray):
            err_type = "Expected FloatArray, got {}".format(type(floatarray))
            raise TypeError(err_type)
        if floatarray.array.dtype in (np.float32,):
            length, d = 32, np.uint32
        elif floatarray.array.dtype in (np.float64, float,):
            length, d = 64, np.uint64
        else:
            err_msg = 'Expected 32 or 64 bits, got {}'.format(floatarray.array.dtype)
            raise TypeError(err_msg)
        data = _vlindstrom(floatarray.array, length).astype(d)
        return IntegerArray(data)


def _lindstrom(value, length):
    result = ba(floatbe=value, length=length) ^ ba(floatbe=2, length=length)
    if value < 0:
        result.invert()
    else:
        result.invert(0)
    return result.uintbe
    # return result.U
_vlindstrom = np.frompyfunc(_lindstrom, 2, 1)
