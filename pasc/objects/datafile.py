#!/usr/bin/env python
# coding: utf-8
"""Classes of Datafiles Arrays."""

import os
import xarray as xr
from pasc.backend import BaseData
from pasc.objects.floatarray import FloatArray


class Datafile(BaseData):

    def __init__(self, data):
        self.data = data

    def _get_data(self):
        return self._data
    def _set_data(self, value):
        if isinstance(value, xr.Dataset):
            self._data = value
        elif isinstance(value, xr.DataArray):
            self._data = value.to_dataset()
        elif isinstance(value, (str, bytes)) and os.path.isfile(value):
            self._data = self.from_netcdf(value).data
        else:
            err_type = "Expected xr.Dataset, got {}".format(type(value))
            raise TypeError(err_type)
    data = property(_get_data, _set_data)

    def to_floatarray(self, var):
        dataarray = getattr(self.data, var)
        return FloatArray(dataarray.values)

    @staticmethod
    def from_netcdf(filename):
        return Datafile(xr.open_dataset(filename))
