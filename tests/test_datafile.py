#!/usr/bin/env python
# coding: utf-8
"""Tests for datafile objects."""

from pkg_resources import resource_filename as resource
import pytest
import numpy as np
from pasc.objects import datafile, floatarray
from pasc.toolbox import load_data

FILENAME = __file__
NC_DATAFILENAME = resource('pasc', 'data/sresa1b_ncar_ccsm3-example.nc')
VALID_INIT = [load_data('pre'), load_data('pre').tas, NC_DATAFILENAME]
INVALID_INIT = [232, np.arange(42), "asdad"]


def test_err_not_netcdffile():
    with pytest.raises(OSError) as err:
        _ = datafile.Datafile(FILENAME)
    assert "Unknown" in str(err)


@pytest.mark.parametrize('dinit', VALID_INIT)
def test_valid_initialisation(dinit):
    init = datafile.Datafile(dinit)
    assert isinstance(init, datafile.Datafile)


@pytest.mark.parametrize('dinit', INVALID_INIT)
def test_invalid_initialisation(dinit):
    with pytest.raises(TypeError) as err:
        _ = datafile.Datafile(dinit)
    assert "Expected xr.Dataset" in str(err)


def test_to_floatarray():
    df = datafile.Datafile(load_data('pre'))
    result = df.to_floatarray(var='tas')
    assert isinstance(result, floatarray.FloatArray)
    assert result == floatarray.FloatArray(load_data('pre').tas)


def test_from_file():
    d = datafile.Datafile.from_netcdf(NC_DATAFILENAME)
    assert np.array_equal(d.data, datafile.Datafile(NC_DATAFILENAME).data)
    assert np.array_equal(d.data, load_data('pre'))
    assert d == load_data('pre')
