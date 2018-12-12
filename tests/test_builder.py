#!/usr/bin/env python
# coding: utf-8
"""Tests for Base Builder modifiers."""

from pasc.toolbox import get_data_path as gdp, generateIndexArray as gia
from pasc.toolbox import flood as fl

from pasc.modifier.mapper import RawBinary
from pasc.modifier.sequencer import Linear
from pasc.objects.datafile import Datafile

from pasc.modifier import builder as Bld
from pasc.objects.floatarray import FloatArray
from pasc.objects.integerarray import IntegerArray as IA  # Input
from pasc.objects.informationspace import InformationSpace as IS  # Output

import pytest
import numpy as np

path = gdp('pre')
farr = Datafile(path).to_floatarray('tas')
iarr = RawBinary.map(farr)
seq = Linear.flatten(startnode=12, integerarray=iarr)

BUILDERS = [
    Bld.Builder,
    Bld.RestrictedBuilder(1),
]


@pytest.fixture(scope="session")
def valids():
    # the returned fixture value will be shared for
    # all tests needing it
    t = gia((3, 4))
    t.flat[0] = fl.getNAN(t.dtype)
    t.flat[-1] = fl.getNAN(t.dtype)
    space = Bld._buildInfoSpace(t, 5)
    exp = {1: [t[:, 1:2].squeeze(), t[1:2, :].squeeze()],
           0: [t[1:2, 1:2].squeeze(axis=0)],
           2: [t[1:, :-1], t[:-1, 1:], t[:, 1:-1]]}
    return [(space, exp), ]


def test_generate_testcases_and_expected(valids):
    case, expected = valids[0]
    for k, v in case.items():
        assert len(v) == len(expected[k])
        for arr in expected[k]:
            assert np.sum([np.array_equal(arr, x) for x in v]) == 1, arr


@pytest.mark.parametrize('builder', BUILDERS)
def test_valid(builder):
    space = builder.build_informationspace(seq)
    assert isinstance(space, IS)


def test_valid_fail(valids):
    # TODO: define tests specific for RestrictedBuilder
    pass

@pytest.mark.parametrize('builder', [Bld.RestrictedBuilder(5), Bld.Builder])
def test_search_by_index(builder):
    error = np.arange(9, dtype=float).reshape(3,3)
    error[-1] = 7
    iarr = RawBinary.map(FloatArray.from_numpy(error))
    seq = Linear.flatten(0, iarr)
    space = builder.build_informationspace(seq)
    assert isinstance(space, IS)
