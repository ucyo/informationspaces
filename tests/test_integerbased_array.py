#!/usr/bin/env python
# coding: utf-8
"""Test cases for instances of array objects."""

import pytest
import numpy as np
from pasc.objects.residualarray import ResidualArray
from pasc.objects.integerarray import IntegerArray
from pasc.objects.predictionarray import PredictionArray

ARR = np.arange(34, dtype=np.int32)
DTYPES = [np.float32, float, "str"]
INPUTS = [2, "sdf", (324, 122), [231.1, "weq"]]
ARRAYS = [ResidualArray, IntegerArray, PredictionArray]

def test_importtype():
    """"Correct initialisation."""
    assert isinstance(ResidualArray(ARR), ResidualArray)


@pytest.mark.parametrize("form", DTYPES)
@pytest.mark.parametrize("array", ARRAYS)
def test_raise_error(array, form):
    """Wrong np.dtypes of array."""
    with pytest.raises(TypeError) as err:
        array(ARR.astype(form))
    assert "Expected np.nd" in str(err)


@pytest.mark.parametrize("value", INPUTS)
@pytest.mark.parametrize("array", ARRAYS)
def test_value_error(array, value):
    """Wrong input types."""
    with pytest.raises(TypeError) as err:
        array(value)
    assert "Expected np.nd" in str(err)


def test_equal_floatarray():
    """Comparison with itself."""
    assert ResidualArray(ARR) == ResidualArray(ARR)


def test_floatarray_with_npndarray():
    """Comparison with related values."""
    assert ResidualArray(ARR) == ARR


@pytest.mark.parametrize("value", INPUTS)
@pytest.mark.parametrize("array", ARRAYS)
def test_floatarray_with_other_object(array, value):
    """Comparison with arbitary types."""
    with pytest.raises(TypeError) as err:
        _ = array(ARR) == value
    assert "Comparison failed" in str(err)
