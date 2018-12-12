#!/usr/bin/env python
# coding: utf-8
"""
Reader for transformation of different kind of files to FloatArray with
the ability to choose subsets of the source data.
"""

import os
import xarray as xr
from pasc.objects import floatarray as fl
import numpy as np
from pasc.toolbox import get_data_path


class Reader:

    name = "Reader"

    @staticmethod
    def from_dataarray(dataarray):#, size=None, seed=None, error=.05, *args, **kwargs):
        dataarray = _raiseTypeError(dataarray, xr.DataArray)
        # if size is not None:
        #     dataarray = _chooseRandomSubset(
        #         dataarray, size=size, seed=seed, error=error)
        dataarray = dataarray.astype(np.float32)  # TODO: Forced to do only 32 bits
        result = fl.FloatArray(dataarray.values)
        return result

    @staticmethod
    def from_dataset(dataset, var):#, size=None, seed=None, error=.05, *args, **kwargs):
        dataset = _raiseTypeError(dataset, xr.Dataset)
        if not hasattr(dataset, var):
            err = "{} not in Dataset".format(var)
            raise KeyError(err)
        dataarray = getattr(dataset, var)
        return Reader.from_dataarray(dataarray=dataarray)#, size=size, seed=seed, error=error)

    @staticmethod
    def from_numpy(array):#, size=None, seed=None, error=.05, *args, **kwargs):
        array = _raiseTypeError(array, np.ndarray)
        if array.dtype not in (float, np.float32, np.float64):
            err = "Expected float dtype, got {}".format(array.dtype)
            raise TypeError(err)
        # if size is not None:
        #     array = _chooseRandomSubsetND(arr=array, size=size, seed=seed, error=error)
        array = array.astype(np.float32)  # TODO: Forced to do only 32 bits
        return fl.FloatArray(array)

    @staticmethod
    def from_netcdf(filename, var, *args, **kwargs):
        if not os.path.isfile(filename):
            err = "{} is not a file.".format(filename)
            raise FileNotFoundError(err)
        ds = xr.open_dataset(filename, *args, **kwargs)
        return Reader.from_dataset(dataset=ds, var=var)#, size=size, seed=seed, error=error)

    @staticmethod
    def from_data(key, var, *args, **kwargs):#, size=None, seed=None, error=.05, *args, **kwargs):
        path = get_data_path(key)
        return Reader.from_netcdf(path, var,  *args, **kwargs)#, size, seed, error, *args, **kwargs)


def _raiseTypeError(obj, clas):
    if not isinstance(obj, clas):
        err = "Expected {}, got {}".format(clas, type(obj))
        raise TypeError(err)
    return obj
