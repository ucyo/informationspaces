#!/usr/bin/env python
# coding: utf-8
"""
Change directions of files/input.
"""
import xarray as xr
import numpy as np
from pasc.objects.floatarray import FloatArray


class Transposer:

    name = "Transposer"

    @staticmethod
    def transpose(obj, *order):
        if isinstance(order[0], tuple):
            order = order[0]
        if isinstance(obj, np.ndarray):
            newobj = transposeND(obj, order)
        elif isinstance(obj, xr.DataArray):
            newobj = transposeDA(obj, order)
        elif isinstance(obj, FloatArray):
            tmp = transposeND(obj.array, order)
            newobj = FloatArray(tmp)
        else:
            raise TypeError("Can't understand datatype of obj.")
        return newobj


def transposeND(arr, order):
    return np.transpose(arr, order)


def transposeDA(daarr, order):
    if all([(isinstance(x, int) and x < daarr.ndim) for x in order]):
        idxtable = {k: v for k, v in zip(range(daarr.ndim), daarr.dims)}
        order = [idxtable[x] for x in order]
        result = daarr.transpose(*order)
    elif all([(isinstance(x, str) and x in daarr.dims) for x in order]):
        result = daarr.transpose(*order)
    else:
        raise TypeError("Can not understand {}".format(order))
    return result


if __name__ == '__main__':
    from pasc.toolbox import get_data_path

    ds = xr.open_dataset(get_data_path('pre'))
    dsa = Transposer.transpose(ds.tas, 'lon', 'time', 'lat')
    print("Before \t{}\nAfter \t{}".format(ds.tas.dims, dsa.dims))
