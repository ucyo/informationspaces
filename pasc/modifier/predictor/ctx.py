#!/usr/bin/env python
# coding: utf-8
"""Predictor functions using Hashing and VHT/VPT tables for prediction."""

from functools import partial
from collections import namedtuple
from pasc.modifier.predictor.core import Stride
from pasc.toolbox.context import ContextHash, Select
from pasc.toolbox.flood import getNAN
import numpy as np
from scipy import linalg


class ContextHashPredictor:

    name = 'Context Hash Predicto (Base)'

    def __init__(self, vht=1, vpt=1, bits=32, *args, **kwargs):
        self.bits = bits
        self.vht = np.ones(vht) * np.nan
        self.vpt = np.ones(vpt) * np.nan

    def update(self, val):
        raise NotImplementedError("Update not implemented")

    def predict(self):
        raise NotImplementedError("Predict not implemented")

    def _add_val_vht(self, val):
        """Add value to beginning of table and kick last one out."""
        self.vht = np.concatenate([[val], self.vht[:-1]])


class Ratana(ContextHashPredictor):
    """
    Predictor defined in Ratanaworabhan et al. 2006 (adjusted to 32 Bits).

    Note
    ====
    The actual paper describes a 64 Bit algorithm. For this algorithm the
    algorithm has been adjusted to work with 32 Bits.

    Changes to original paper
    =========================
    - First select from '<14' to '<10'
    - Last select from '<20' to '<12'
    """

    name = 'Ratana'

    def __init__(self, order, *args, **kwargs):
        super(Ratana, self).__init__()
        self.order = order
        self.lastvalue, self.vht = 0, [None] * self.order
        self.vpt = dict()
        self.default = Stride()
        self.contexthash = ContextHash(
            R=['<10' for x in range(self.order)],
            F=[None] * self.order,
            S=['<' + str(x * 5) for x in range(self.order)],
            L='>12', bits=32
        )

    def _hashfunction(self):
        idx = self.contexthash(self.vht)
        return idx

    def predict(self):
        idx = self._hashfunction()
        result = False
        if idx is not None and idx in self.vpt.keys():
            predVPT = self.vpt[idx].prediction
            if predVPT is not None:
                result = predVPT
        self._idx = idx
        self._newidx = True
        return result if result else self.default.predict()

    def update(self, val):
        if not self._newidx:
            raise Exception('Prediction must be executed before update')
        idx = self._idx
        self._newidx = False
        truediff = self.lastvalue - val if self.lastvalue >= val else val - self.lastvalue
        if idx in self.vpt.keys():
            self.vpt[idx] = self.vpt[idx].insert(val)
        elif idx is not None:
            self.vpt[idx] = RatVPT(val, None)

        self.lastvalue = val
        self._add_val_vht(truediff)
        self.default.update(val)

    def __repr__(self):
        return "{} (order: {})".format(self.name, self.order)


Ratana3 = partial(Ratana, order=3)
Ratana5 = partial(Ratana, order=5)

# New formulation of VPT for rata
_RatVPT = namedtuple('Predictor', 'pred1,pred2')


class RatVPT(_RatVPT):

    @property  # staticmethod
    def prediction(self):
        if self.pred1 is None:
            return None
        elif self.pred2 is None:
            return self.pred1
        elif self.notClose:
            return self.pred1
        return self.pred1 + (self.pred1 - self.pred2)

    @property
    def notClose(self):
        first10Bits = Select('<10', 32)
        return first10Bits(self.pred1) != first10Bits(self.pred2)

    def insert(self, value):
        return self.__class__(value, self.pred1)


class PascalLinear(ContextHashPredictor):

    name = 'Pascal (1D)'

    def __init__(self, vht, *args, **kwargs):
        super().__init__(vht, vpt=1, *args, **kwargs)
        self.vht = self.vht.astype(np.int32)
        self.weights = np.insert(_get_pascal_weights(vht), [
                                 0], [0] * vht, axis=0)
        self.nan = getNAN(self.vht.dtype)

    def predict(self):
        vals = self.vht.size - np.sum([x == self.nan for x in self.vht])
        result = sum(self.weights[vals] * self.vht)
        # print(self.vht, result, vals)
        return result

    def update(self, val):
        self._add_val_vht(val)

    def __repr__(self):
        return "{} (vht {})".format(self.name, len(self.vht))


PascalLinear3 = partial(PascalLinear, vht=3)
PascalLinear2 = partial(PascalLinear, vht=2)
PascalLinear1 = partial(PascalLinear, vht=1)
PascalLinear4 = partial(PascalLinear, vht=4)
PascalLinear5 = partial(PascalLinear, vht=5)


def _get_pascal_weights(depth):
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
