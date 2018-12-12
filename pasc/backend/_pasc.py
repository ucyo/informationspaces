#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interface for Pasc objects."""

from abc import ABCMeta, abstractproperty
from pasc.toolbox import check_methods


class PascInterface(metaclass=ABCMeta):
    """Interface for a Pasc file.

    Attributes
    ==========
    header : dict(str)
        Header information needed for decoding file.
    data : binary
        Actual data to be written as binary.
    """

    __slots__ = ()

    @abstractproperty
    def header(self):
        """Header information for pasc file."""
        return dict()

    @abstractproperty
    def data(self):
        """Data of pasc file."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is PascInterface:
            return check_methods(C, "header", "data")
        return NotImplemented
