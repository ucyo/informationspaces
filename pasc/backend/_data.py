#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interface for Netcdf objects."""

from abc import ABCMeta, abstractproperty
from pasc.toolbox import check_methods
import numpy as np

class DataInterface(metaclass=ABCMeta):
    """Wrapper for a Netcdf file.

    Attributes
    ==========
    data : xr.Dataset
        Data to be compressed.
    """

    __slots__ = ()

    @abstractproperty
    def data(self):
        """xr.Dataset of netcdf file."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DataInterface:
            return check_methods(C, "data")
        return NotImplemented


class BaseData(DataInterface):

    def __repr__(self):
        return str(self.data)
