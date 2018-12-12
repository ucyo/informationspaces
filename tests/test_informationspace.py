#!/usr/bin/env python
# coding: utf-8
"""Tests for Information Space."""

from pasc.backend import BaseInformationContext as bIC
from pasc.objects.informationspace import InformationSpace as IS
from pasc.objects.informationcontext import InformationContext as IC
from pasc.toolbox import generateRandomIndexArray as gria
import numpy as np
import pytest


VALID = [
    {0: IC([np.array([21], dtype=int), np.array([41], dtype=int)]),
     1: IC([gria((35,), 1, seed=31), gria((11,), 1, seed=231)]),
    },
    {0: [np.array([21], dtype=int), np.array([41], dtype=int)],
     1: [gria((35,), 1, seed=31), gria((11,), 1, seed=231)],
     }
]

INVALID = [
    (TypeError, "Expected dict", [3124, 3123]),
    (TypeError, "Expected IC", {"sadfa":"asdfa"})
]


@pytest.mark.parametrize("dicts", VALID)
def test_valid_input(dicts):
    space = IS(dicts)
    assert isinstance(space, IS)
    assert space[1].dims == 1
    assert all([isinstance(x, bIC) for x in space.space.values()])


@pytest.mark.parametrize("err,msg,dicts", INVALID)
def test_invalid_input(err, msg, dicts):
    with pytest.raises(err) as m:
        _ = IS(dicts)
    assert msg in str(m)
