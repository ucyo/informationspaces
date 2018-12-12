#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interface for Coded array objects."""

from abc import ABCMeta, abstractproperty
from pasc.toolbox import check_methods


class CodedInterface(metaclass=ABCMeta):
    """Coded object ready to be written on disk.

    The decode attribute can be anyy kind of information needed to extract
    the residual array from coded array.

    Attributes
    ==========
    array : list/np.array
        Actual values to be saved on disk
    decode : dict(str)
        Information needed for decompression
    """

    __slots__ = ()

    @abstractproperty
    def array(self):
        """Actual bits to be saved on disk."""
        return None

    @abstractproperty
    def decode(self):
        """Actual bits to be saved on disk."""
        return dict()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is CodedInterface:
            return check_methods(C, "array", "decode")
        return NotImplemented
