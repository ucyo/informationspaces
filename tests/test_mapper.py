#!/usr/bin/env python
# coding: utf-8
"""Testing mapper functions."""

from pasc.modifier import mapper
from pasc.objects.floatarray import FloatArray
from pasc.objects.integerarray import IntegerArray
import numpy as np
import pytest


MAPPER = [
    mapper.Lindstrom,
    mapper.RawBinary,
    ]

INPUTGROUP_VALID = [
    FloatArray(np.arange(24, dtype=np.float32)),
    ]

INPUTGROUP_INVALID = [
    IntegerArray(np.arange(24, dtype=np.int32)),
    ]


@pytest.mark.parametrize('mapmod', MAPPER)
@pytest.mark.parametrize('inputgroup', INPUTGROUP_VALID)
def test_map_function_output(inputgroup, mapmod):
    """Result of Mapper is an IntegerArray using map method."""
    mapfunc = mapmod.map(inputgroup)
    assert isinstance(mapfunc, IntegerArray)

# Call function feature will not be implemented (yet)
#
# @pytest.mark.parametrize('mapper', MAPPER)
# @pytest.mark.parametrize('inputgroup', INPUTGROUP_VALID)
# def test_call_function_output(inputgroup, mapper):
#     """Result of Mapper is an IntegerArray using call method."""
#     callfunc = mapper(inputgroup)
#     assert isinstance(callfunc, IntegerArray)
#
#
# @pytest.mark.parametrize('mapper', MAPPER)
# @pytest.mark.parametrize('inputgroup', INPUTGROUP_VALID)
# def test_same_output_call_map(inputgroup, mapper):
#     """Result of map and call method are the same."""
#     callfunc = mapper(inputgroup)
#     mapfunc = mapper.map(inputgroup)
#     assert callfunc == mapfunc


@pytest.mark.parametrize('mapper', MAPPER)
@pytest.mark.parametrize('inputgroup', INPUTGROUP_INVALID)
def test_wrong_input_group(inputgroup, mapper):
    """Wrong input throws TypeError."""
    with pytest.raises(TypeError) as err:
        _ = mapper.map(inputgroup)
    assert "Expected FloatArray, got" in str(err)
