#!/usr/bin/env python
# coding: utf-8
"""Types of Information Context."""

import logging
from collections import namedtuple
from pasc.backend import BaseInformationContext
from pasc.toolbox.flood import getNAN
import numpy as np
_log = logging.getLogger(__name__)

CTX = namedtuple("InfoContext", "data, info")


class InformationContext(BaseInformationContext):

    name = "InformationContext"

    def __init__(self, arrs):
        self.context = arrs

    @property
    def dims(self):
        return self.context[0].data.ndim if self.context else None

    def _get_context(self):
        return self._context

    def _set_context(self, value):
        if not all([isinstance(x, CTX) for x in value]):
            err_msg = "All values must be of type {}, got {}".format(
                CTX, [type(x) for x in value])
            raise TypeError(err_msg)
        if not all([isinstance(x.data, np.ndarray) for x in value]):
            err_msg = "All values must be of type np.ndarray, got {}.".format(
                [type(x) for x in value])
            raise TypeError(err_msg)
        if not all([x.data.ndim == value[0].data.ndim for x in value]):
            err_msg = "All arrays must have the same dimension."
            raise ValueError(err_msg)
        for i, arr in enumerate(value):
            if arr.data.size <= 1:
                del value[i]
            elif np.sum([y == getNAN(arr.data.dtype) for y in arr.data]) != 1:
                err_msg = "All arrays must have exactly one(!) NaN value,"\
                    " got in \n{}.".format(arr.data)
                raise ValueError(err_msg)
        self._context = value
    context = property(_get_context, _set_context)

    @staticmethod
    def create(data, **kwargs):
        _log.debug("XX %s %s", data, kwargs)
        if kwargs.get('id', False):
            try:
                kwargs['id'] = setID(*kwargs['id'])
            except TypeError:
                # If 'id' is not a tuple, just take it as is
                _log.warning("Info is not tuple, taking as is: %s %s",
                             data, kwargs)
        data.flags.writeable = False
        return CTX(data, kwargs)


def setID(*dim):

    try:
        k = dim[0]
    except IndexError:
        return 0
    return (1 << k) | setID(*dim[1:])
