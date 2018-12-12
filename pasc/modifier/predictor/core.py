#!/usr/bin/env python
# coding: utf-8
"""Predictor classes."""

from functools import partial
from pasc.backend import CorePredictor
from pasc.toolbox.context import Select


class LastValue(CorePredictor):
    """Last Value Prediction."""

    name = 'Last Value'

    def __init__(self, *args, **kwargs):
        self._prev = 0
        _, _ = args, kwargs

    def update(self, val):
        self._prev = val

    def predict(self):
        return self._prev


class Stride(CorePredictor):

    name = 'Stride'

    def __init__(self, *args, **kwargs):
        self._stride = 0
        self._prev = 0
        _, _ = args, kwargs

    def update(self, val):
        self._stride = val - self._prev
        self._prev = val

    def predict(self):
        return self._prev + self._stride


class TwoStride(CorePredictor):

    name = 'Stride (2)'

    def __init__(self, *args, **kwargs):
        self._bestStride = 0
        self._lastStride = 0
        self._prev = 0
        _, _ = args, kwargs

    def update(self, val):
        new_stride = val - self._prev
        if abs(new_stride - self._lastStride) < abs(new_stride - self._bestStride):
            self._bestStride = new_stride
        self._lastStride = new_stride

    def predict(self):
        return self._prev + self._bestStride


class StrideConfidence(CorePredictor):

    name = 'Stride Conf.'

    def __init__(self, threshold, *args, **kwargs):
        self._confidence = 0
        self._stride = 0
        self._prev = 0
        self._select = Select('<11')
        self._threshold = threshold
        _, _ = args, kwargs

    def update(self, val):
        trueStride = val - self._prev
        self._updateConfidence(trueStride)
        self._confidence = max(0, self._confidence)  # do not allow neg. values
        if self._confidence < self._threshold:
            self._stride = trueStride

    def _updateConfidence(self, trueStride):
        if self._select(trueStride) == self._select(self._stride):
            self._confidence += 1
        else:
            self._confidence -= 2

    def predict(self):
        return self._prev + self._stride

    def __repr__(self):
        return "{} (threshold: {})".format(self.name, self._threshold)
StrideConfidence7 = partial(StrideConfidence, threshold=7)

class Akumuli(CorePredictor):

    name = 'Akumuli'

    def __init__(self, bits=32, table_size=128, *args, **kwargs):
        if (table_size & (table_size - 1)) != 0:
            raise ValueError("`table_size` should be a power of two!")
        self._table = [0]*table_size
        self._mask = table_size - 1
        self._last_hash = 0
        self._bits = bits
        self._fff = 0xFFFFFFFF if bits == 32 else 0xFFFFFFFFFFFFFFFF
        self._ctx = 32 - 11 if bits == 32 else 64-14

    def update(self, val):
        # print('Update', val)
        self._table[self._last_hash] = val
        self._last_hash = self._hashfunction(val)

    def predict(self):
        return self._table[self._last_hash]

    def _hashfunction(self, val):
        # print('HF', [type(x) for x in [self._last_hash, self._fff, val]])
        shifted = (self._last_hash << 5) & self._fff
        result = (shifted ^ (val >> self._ctx)) & self._mask
        return int(result)  # TODO: Somehow a transformation to int is necessary. Why?

    def __repr__(self):
        return "{} (bits {})".format(self.name, self._bits)
