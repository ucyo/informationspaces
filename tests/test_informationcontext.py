#!/usr/bin/env python
# coding: utf-8
"""Tests for informationcontexts."""

from pasc.objects.informationcontext import InformationContext as IC
from pasc.toolbox import generateRandomIndexArray as gria
import numpy as np
import pytest

VALID = [
    [np.array([21], dtype=int), np.array([41], dtype=int)],
    [np.array([], dtype=int)],
    [gria((3, 3), 1), gria((3, 2), 1)],
]

INVALID = [
    (TypeError, "of type np.ndarray", [
        [123, 12, 31, 21, 4], (231, 31, 12312, 123)]),
    (ValueError, "same dimension", [gria((3, 2), 1), gria((3, 2, 2), 1)]),
    (ValueError, "exactly one(!)", [gria((3, 2, 2), 1), gria((3, 2, 2), 2)]),
    (ValueError, "exactly one(!)", [gria((3, 2, 2), 2), gria((3, 2, 2), 4)]),
]


@pytest.mark.parametrize("arrs", VALID)
def test_valid_entries(arrs):
    ic = IC(arrs)
    assert isinstance(ic, IC)
    assert np.array_equal(ic.context[0], arrs[0])
    assert ic.dims == arrs[0].ndim
    assert np.array_equal(ic[0], arrs[0])


@pytest.mark.parametrize("err, msg, arrs", INVALID)
def test_invalid_entries(err, msg, arrs):
    with pytest.raises(err) as m:
        _ = IC(arrs)
    assert msg in str(m)
