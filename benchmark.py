#!/usr/bin/env python
# coding: utf-8
"""Benchmark of predictions."""
import logging
from log import setup_logging
from pasc.objects.floatarray import FloatArray
from pasc.modifier.predictor import mixed
from pasc.modifier.predictor import core
from pasc.modifier.predictor import ctx
from pasc.modifier import mapper as mp, sequencer as sq, builder as bd
from pasc.modifier import subtractor as sb  # , predictor as pd
from pasc.toolbox import feed, qualityassessment as qa
from pasc.toolbox import manager as mgt
setup_logging()
_log = logging.getLogger(__name__)

predictors = [
    core.Akumuli,
    core.LastValue,
    core.Stride,
    core.StrideConfidence7,
    core.TwoStride,
    # ctx.Ratana3,
    # ctx.Ratana5,
    ctx.PascalLinear1,
    ctx.PascalLinear2,
    ctx.PascalLinear3,
    ctx.PascalLinear4,
    ctx.PascalLinear5,
]


mixed = [
    # partial(mixed.MostRight, predictors=predictors),
    # partial(mixed.LastBest, predictors=predictors),
]


def seqPred(seq, *args, **kwargs):
    """Testing of all implemented predictors."""
    print('*' * 27)
    for preds in [predictors, mixed]:
        for predictor in preds:
            try:
                feeder = feed.SeqFeeder(predictor, *args, **kwargs)
                pname, parr = feeder.feed(seq)
                res = sb.XOR.subtract(parr, iarr)
                print("{:27}".format(pname), qa.QA(res))
                s = "seq {}".format(pname)
                s = " ".join(s.split())
                filename = "".join(x if x.isalnum() else "_" for x in s)
                yield filename, qa.QA(res).lzcND
                res.dump('./benchmarkResults/{}.np'.format(filename))
            except (TypeError, NotImplementedError):
                print("{:27}".format(predictor.name), 'err')
                raise
        print('*' * 27)


managers = [
    # mgt.CountManager,  # Count is not deterministic
    mgt.AverageManager,
    mgt.MinManager,
    mgt.MaxManager,
    mgt.ReproduceManager,
    mgt.LastBestManager,
]


def spacePred(seq, *args, **kwargs):
    """
    Testing of all implemented predictors using Managers.
    """
    print('*' * 27)
    for manager in managers:
        print("{:27}".format(manager.name))
        for preds in [predictors, mixed]:
            for predictor in preds:
                m = manager(predictor, vptpow=12)
                try:
                    feeder = feed.SpaceFeeder1DMA("")
                    agg = mgt.Aggregate(feeder, m, 1, **kwargs)
                    pname, parr = agg.feed(seq)
                    res = sb.XOR.subtract(parr, iarr)
                    print("{:27}".format(pname), qa.QA(res))
                    s = "space {} {}".format(m, pname)
                    s = " ".join(s.split())
                    filename = "".join(x if x.isalnum() else "_" for x in s)
                    yield filename, qa.QA(res).lzcND
                    res.reshape(seq.shape).dump(
                        './benchmarkResults/{}.np'.format(filename))
                except (TypeError, NotImplementedError):
                    print("{:27}".format(predictor.name), 'err')
                    raise
            print('*' * 27)


if __name__ == '__main__':
    import xarray as xr
    import numpy as np
    farr = FloatArray.from_data('pre', 'tas', decode_times=False)
    # seqstart = 0
    seqstart = 350
    # sequencer = sq.Linear
    sequencer = sq.BlossomC
    # sequencer = sq.ChequerboardC
    # sequencer = sq.BlockC
    x, y = 32, 32
    farr = FloatArray.from_numpy(farr.array[0, 32:32 + x, :y].T)
    mapper = mp.RawBinary
    iarr = mapper.map(farr)

    ds = xr.Dataset({'map': (['x', 'y'], iarr.array)},
                    coords={'x': np.arange(x),
                            'y': np.arange(y)})
    ds.attrs['mapper'] = mapper.name
    ds.attrs['sequencer'] = sequencer.name
    ds.attrs['startidx'] = seqstart

    seq = sequencer.flatten(seqstart, iarr)
    args = ()
    kwargs = dict(restriction=2)
    _log.debug("SeqIdx: %s - SeqDat: %s - SeqShape: %s",
               seq.sequence, seq.data, seq.shape)
    for fname, parr in seqPred(seq, *args, **kwargs):
        ds[fname] = xr.DataArray(parr, dims=['x', 'y'])
    for fname, pa in spacePred(seq, *args, **kwargs):
        ds[fname] = xr.DataArray(pa, dims=['x', 'y'])
    ds['origin'] = xr.DataArray(farr.array, dims=['x', 'y'])
    ds.to_netcdf(
        './benchmarkResults/results_{}_{}_{}_mgt_T.nc'.format(mapper.name,
                                                              sequencer.name,
                                                              seqstart))

    # python -m cProfile -o benchmark.profile benchmark.py
    # cprofilev -f benchmark.profile -p 4001
