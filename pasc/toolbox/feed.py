#!/usr/bin/env python
# coding: utf-8
"""
Feeder prepares the Sequence or Information Space objects to be consumed
by the Predictors. This step is necessary to separate the definition of
the predictor from data consumption. Therefore the predictor only needs
to be aware of the next value and update itself.
"""
import logging
from pasc.backend import BaseInformationSpace  # Input
from pasc.backend import BaseSequence  # Input
from pasc.objects.predictionarray import PredictionArray  # Output
# from pasc.objects.sequence import IndexSequence
from pasc.modifier import builder as bd
from pasc.toolbox import get_bits
from pasc.toolbox.flood import getNAN
import numpy as np
_log = logging.getLogger(__name__)


def _check_input(obj, source, dim=None):
    if not isinstance(obj, source):
        err = "Expected {} object, got {}".format(source, type(obj))
        raise TypeError(err)
    if isinstance(obj, BaseInformationSpace) and dim is None and obj.space.get(dim, False):
        err = "Dimension {} is not present/empty.".format(dim)
        raise TypeError(err)
    return obj


class BaseFeeder:
    """Feeder class as a common base for following feeder methods.
    """

    def __init__(self, predictor, *args, **kwargs):
        self.pred = predictor
        self.args = args
        self.kwargs = kwargs
        self.obj = 0  # Count elements predicted

    def reset(self):
        """Resetting the Predictor to initial state.

        Possibility for the Feeder entity to reset the predictor to its original initial state.
        """
        self.predictor = self.pred(*self.args, **self.kwargs)
        self.obj = 0

    def step(self, value):
        """Predicting next element and updating with actual value.
        """
        prediction = self.predictor.predict()
        self.predictor.update(value)
        _log.info("Obj: %s - Seq(%s): %s", self.obj,
                  self.predictor, prediction)
        self.obj += 1
        return prediction


class SeqFeeder(BaseFeeder):
    """Feeder for predictors using the Sequence objects for prediction."""

    def feed(self, seqobj, pa=True):
        """Feed Sequence object to initial predictors"""
        seqobj = _check_input(seqobj, BaseSequence)
        self.kwargs['bits'] = get_bits(seqobj.data)

        self.reset()
        if not pa:
            result = np.array([self.step(x) for x in seqobj.data])
        else:
            result = np.zeros_like(seqobj.data).reshape(seqobj.shape)
            for i in range(len(seqobj.sequence)):
                idx = seqobj[i]
                true = seqobj.data[i]
                prediction = self.step(true)
                result.flat[idx] = prediction
                _log.debug(
                    "Idx: %s - Seq(%s): %s [Truth: %s]", idx, self.predictor, prediction, true)
            result = PredictionArray(result)
        name = str(self.predictor)
        self.reset()
        return name, result


class SpaceFeederGen(BaseFeeder):

    def __init__(self, builder, *args, **kwargs):
        self.builder = bd.GeneralBuild
        self.args = args
        self.kwargs = kwargs
    #
    # def reset(self):
    #     self.builder = self.build(*self.args, **self.kwargs)

    def feed(self, seq, restriction):
        seq = _check_input(seq, BaseSequence)
        nan = getNAN(seq.data.dtype)
        fillvalue = np.max(seq.data) + 1
        arr = np.ones(shape=seq.shape, dtype=seq.dtype) * nan
        # self.reset()
        for i, searchidx in enumerate(seq.sequence):
            truth = seq.data[i]
            arr.flat[searchidx] = fillvalue
            yield truth, self.builder.build_infospace(arr, searchval=fillvalue, restriction=restriction)
            arr.flat[searchidx] = truth


class SpaceFeeder1DMA(BaseFeeder):

    def __init__(self, builder, *args, **kwargs):
        self.builder = bd.OneDBuildMA
        self.args = args
        self.kwargs = kwargs

    def feed(self, seq, restriction):
        seq = _check_input(seq, BaseSequence)
        ma = np.ma.ones(seq.shape).astype(seq.dtype)
        for i, k in zip(seq.sequence, seq.data):
            ma.data.flat[i] = k
        ma.mask = True
        for s in seq.sequence:
            ma.mask.flat[s] = False
            ispace = self.builder.build_infospace(
                ma, searchidx=s, restriction=restriction)
            yield ma.data.flat[s], ispace


class _SpaceFeeder_old:

    def __init__(self, builder, *args, **kwargs):
        self.build = builder
        self.args = args
        self.kwargs = kwargs

    def reset(self):
        self.builder = self.build(*self.args, **self.kwargs)

    def feed(self, seqobj):
        seqobj = _check_input(seqobj, BaseSequence)
        self.reset()

        for i in range(1, seqobj.sequence.size):
            t = seqobj.__class__(
                seqobj.sequence[:i], seqobj.shape, seqobj.data[:i])
            yield self.builder.build_informationspace(t)
        self.reset()


if __name__ == '__main__':
    from itertools import islice
    from pasc.objects.floatarray import FloatArray
    from pasc.modifier import mapper as mp, sequencer as sq, builder as bd
    farr = FloatArray.from_data('pre', 'tas', decode_times=False)
    farr = FloatArray.from_numpy(farr.array[0, ::, ::])
    iarr = mp.RawBinary.map(farr)
    seq = sq.Linear.flatten(0, iarr)
    builder = bd.RestrictedBuilder
    kwargs = dict(restriction=5)

    feeder = SpaceFeeder1DMA(builder, **kwargs)
    result = islice(feeder.feed(seq, restriction=4), 902)
    for IS in result:
        print(IS, end='\n\n\n\n')
