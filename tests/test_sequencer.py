#!/usr/bin/env python
# coding: utf-8
"""Tests for Sequencer objects."""

import pytest
import numpy as np
from pasc.objects.integerarray import IntegerArray
from pasc.objects.sequence import Sequence
from pasc.modifier.sequencer import Linear, Block
from pasc.modifier.sequencer import Chequerboard, Blossom


SEQUENCER = {
    'lin': Linear,
    'blo': Block,
    'chq': Chequerboard,
    'bls': Blossom
}


VALID_INPUT = [
    # TODO: Special cases for Chequerboard...
    # IntegerArray(np.arange(14).reshape(2,7)),
    # IntegerArray(np.arange(31)),
    IntegerArray(np.arange(64).reshape(8, 8)),
]


@pytest.mark.parametrize("integerarray", VALID_INPUT)
@pytest.mark.parametrize("sequencer", SEQUENCER.values())
def test_with_correct_input_output(sequencer, integerarray):
    startindex = 1
    seq = sequencer.flatten(startnode=startindex, integerarray=integerarray)
    assert isinstance(seq, Sequence)
    assert seq.sequence[0] == 1


SHAPES = {
    '1d': (27,),
    '3ds': (3, 3, 3),
    '2d': (8, 8),
    '4d': (6, 3, 9, 3),
}
DATA = {k: np.arange(np.prod(x)).reshape(x) for k, x in SHAPES.items()}
INTEGERARRAYS = {k: IntegerArray(arr) for k, arr in DATA.items()}
ORDER = {
    '1d': (0,),
    '3ds': (2, 1, 0),
    '2d': (1, 0),
    '4d': (1, 3, 0, 2),
}
EXPECTED_LINEAR = {
    '1d': np.roll(np.arange(27), -1),
    '3ds': np.roll(np.arange(np.prod(SHAPES['3ds'])).reshape(SHAPES['3ds']).transpose(2, 1, 0).flatten('C'), -9),
    '2d': np.roll(np.arange(np.prod(SHAPES['2d'])).reshape(SHAPES['2d']).flatten('F'), -8),
    '4d': np.roll(np.arange(np.prod(SHAPES['4d'])).reshape(SHAPES['4d']).transpose((1, 3, 0, 2)).flatten('C'), -54),
}


@pytest.mark.parametrize('ival', SHAPES.keys())
def test_expected_Linear(ival):
    seq = Linear.flatten(1, INTEGERARRAYS[ival], ORDER[ival])
    assert np.array_equal(seq.sequence, EXPECTED_LINEAR[ival])
