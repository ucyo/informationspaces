#!/usr/bin/env python
# coding: utf-8
"""Sequencer modifier."""

from pasc.backend import BaseSequencer
from pasc.objects.integerarray import IntegerArray  # Input
from pasc.objects.sequence import IndexSequence  # Output
# from pasc.toolbox import graphtheory as gt
from pasc.toolbox import bfs
import numpy as np


# TODO: Allow only idx as input. Not 'nodenames'.
def _check_input(startnode, integerarray):
    if not isinstance(integerarray, IntegerArray):
        err_msg = "Expected IntegerArray, got {}".format(type(integerarray))
        raise TypeError(err_msg)
    if isinstance(startnode, int):
        startnode = str(startnode + 1)  # Node was given as 'idx'
    return startnode, integerarray


class Linear(BaseSequencer):
    """Linear output sequence for N dimensional arrays."""

    name = 'Linear'

    @staticmethod
    def flatten(startnode, integerarray, order=None):
        startnode, integerarray = _check_input(startnode, integerarray)
        shape = integerarray.array.shape
        tmpdata = np.arange(np.prod(shape)).reshape(shape)
        nodenames = (tmpdata + 1).astype(str)
        if not order or order in ('c', 'C'):
            new = nodenames.copy()
        elif order in ('f', 'F'):
            new = np.transpose(nodenames).copy()
        else:
            new = np.transpose(nodenames, order).copy()

        startidx = np.where(new.flat == startnode)[0][0]
        nodelist = np.roll(new.flat, -startidx)#.flatten()#.tolist()
        seq = nodelist.astype(np.int32) - 1
        data = np.array([integerarray.array.flat[x] for x in seq],
                        dtype=integerarray.array.dtype)
        return IndexSequence(seq, shape, data, order=order)


class Chequerboard(BaseSequencer):
    """Sequencer using a chequerboard style traversal."""

    name = 'Chequerboard'

    @staticmethod
    def flatten(startnode, integerarray, weights=None):
        if not weights:
            _cheq_weights = {
                (0, 1, -1): 2,
                (0, 1, 1): 3,
                (0, -1, 1): 4,
                (0, -1, -1): 5,
            }
            weights = bfs.pack_weights(_cheq_weights)
        startnode, integerarray = _check_input(startnode, integerarray)
        shape = integerarray.array.shape

        _, seq = bfs.BFSCheq(shape=shape, startidx=int(startnode)-1, weights=weights)
        data = np.array([integerarray.array.flat[x] for x in seq])
        return IndexSequence(list(seq), shape, data)


class ChequerboardC(BaseSequencer):
    """Easy Chequerboard without weights.

    Weights argument will be ignored.
    """

    name = "Chequerboard (no weights)"

    @staticmethod
    def flatten(startnode, integerarray, weights=None):
        _ = weights
        startnode, integerarray = _check_input(startnode, integerarray)
        shape = integerarray.shape
        if len(shape) == 1:
            raise ValueError("Incorrect input")
        seq = bfs.CheqNoWeights(shape=shape, startidx=int(startnode)-1)
        seq = list(seq)
        data = np.array([integerarray.array.flat[x] for x in seq])
        return IndexSequence(seq, shape, data)


class BlockC(BaseSequencer):
    """Easy Block without weights.

    Weights argument will be ignored.
    """

    name = "Block (no weights)"

    @staticmethod
    def flatten(startnode, integerarray, weights=None):
        _ = weights
        startnode, integerarray = _check_input(startnode, integerarray)
        shape = integerarray.shape
        seq = bfs.BlocNoWeights(shape=shape, startidx=int(startnode)-1)
        seq = list(seq)
        data = np.array([integerarray.array.flat[x] for x in seq])
        return IndexSequence(seq, shape, data)

class BlossomC(BaseSequencer):
    """Easy Blossom without weights.

    Weights argument will be ignored.
    """

    name = "Blossom (no weights)"

    @staticmethod
    def flatten(startnode, integerarray, weights=None):
        _ = weights
        startnode, integerarray = _check_input(startnode, integerarray)
        shape = integerarray.shape
        seq = bfs.BlosNoWeights(shape=shape, startidx=int(startnode)-1)
        seq = list(seq)
        data = np.array([integerarray.array.flat[x] for x in seq])
        return IndexSequence(seq, shape, data)

class Block(BaseSequencer):
    """Sequencer using a block style traversal."""

    name = "Block"

    @staticmethod
    def flatten(startnode, integerarray, weights=None):
        default_weights = ()
        if not weights:
            weights = default_weights
        startnode, integerarray = _check_input(startnode, integerarray)
        shape = integerarray.array.shape

        _, seq = bfs.BFSBloc(shape=shape, startidx=int(startnode)-1, weights=weights)
        data = np.array([integerarray.array.flat[x] for x in seq])
        return IndexSequence(list(seq), shape, data)

class Blossom(BaseSequencer):
    """Sequencer in the shape of a blossom."""

    name = "Block"

    @staticmethod
    def flatten(startnode, integerarray, weights=None):
        default_weights = ()
        if not weights:
            weights = default_weights
        startnode, integerarray = _check_input(startnode, integerarray)
        shape = integerarray.array.shape

        _, seq = bfs.BFSBlos(shape=shape, startidx=int(startnode)-1, weights=weights)
        data = np.array([integerarray.array.flat[x] for x in seq])
        return IndexSequence(list(seq), shape, data)


if __name__ == '__main__':
    from pasc.objects.floatarray import FloatArray
    from pasc.modifier.mapper import RawBinary
    farr = FloatArray.from_data('pre', 'tas')
    # farr = FloatArray.from_numpy(np.arange(43, dtype=float))
    farr = FloatArray.from_numpy(farr.array[::, ::, ::])
    iarr = RawBinary.map(farr)
    startidx = 0
    print(ChequerboardC.name, ChequerboardC.flatten(startidx, iarr).sequence)
    print(BlossomC.name, BlossomC.flatten(startidx, iarr).sequence)
    print(BlockC.name, BlockC.flatten(startidx, iarr).sequence)
    print(Linear.name, Linear.flatten(startidx, iarr).sequence)
    print(Blossom.name, Blossom.flatten(startidx, iarr).sequence)
    print(Block.name, Block.flatten(startidx, iarr).sequence)
    print(Chequerboard.name, Chequerboard.flatten(startidx, iarr).sequence)
