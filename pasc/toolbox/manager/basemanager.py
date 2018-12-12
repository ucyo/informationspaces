#!/usr/bin/env python
# coding: utf-8
"""
A manager for weighing different predictors based in the appropiate
information space being used.
"""
import logging
from abc import abstractmethod, ABCMeta
import numpy as np
from pasc.objects.predictionarray import PredictionArray
from pasc.toolbox.flood import _wrapper
from pasc.modifier.predictor import core
_log = logging.getLogger(__name__)

DEFAULT = core.LastValue


class BaseManager(metaclass=ABCMeta):

    name = 'Base Manager'

    def __init__(self, predictor, vptpow, *args, **kwargs):
        """Initialisation of general set-up for Information Space prediction.

        Arguments
        =========
        predictor : Predictor
            Predictor to be used.
        vptpow: int
            Power of two for the size of the Value Prediction Table (VPT)
        args, kwargs : .
            Arguments for initialisation of predictor
        """
        self.vpt = dict()
        self.pred = predictor
        self.vptpow = 2**vptpow
        self.args = args
        self.kwargs = kwargs

        self.obj = 0  # Count elements predicted
        self.predictor = self.pred(*args, **kwargs)
        self.default = DEFAULT()

    def reset(self):
        """Reset VPT and predictor."""
        self.vpt = dict()
        self.obj = 0
        self.predictor = self.pred(*self.args, **self.kwargs)
        self.default = DEFAULT()

    def predict(self, ctx):
        """Give a prediction for certain context.

        Predict returns a prediction from the `predictor` for the given context.
        First a key from the known part of the context will be calculated. It will
        be looked if this key has ever occurred in the past (is in VPT). If so,
        that predictor will be used. If not, a clean one will be spawned.
        """
        # key = hash(_wrapper(ctx.data[:-1]))  # % self.vptpow
        # predictor = self.vpt.get(key, self.pred(*self.args, **self.kwargs))
        predictor = self.searchforvpt(ctx)
        result = predictor.predict()
        _log.debug("Predicting | Obj: %s , Ctx: %s > %s",
                   self.obj, ctx, result)
        return result

    def update(self, truth, ctx):
        """Given the truth value and the context the predictor will be updated.

        As in the `self.predict` first a key from the known part of the context is
        being calculated. The predictor at this position in the VPT will be removed
        (if any is there) or a new one spawned. The Predictor gets normally updated.
        The context gets updated via truth value and new key will be generated. The
        predictor then saved based on the key of the new context.
        """
        # key = hash(_wrapper(ctx.data[:-1]))  # % self.vptpow
        # predictor = self.vpt.get(key, self.pred(*self.args, **self.kwargs))
        predictor = self.searchforvpt(ctx)
        predictor.update(truth)
        arr = np.concatenate([ctx.data[1:-1], [truth]])
        newkey = hash(_wrapper(arr))  # % self.vptpow
        _log.debug("Update(%s) - Obj: %s - Ctx: %s - NewCtx: %s|%s - Truth: %s",
                   predictor, self.obj, ctx.data[:-1], arr, newkey, truth)
        self.vpt[newkey] = predictor
        self.default.update(val=truth)
        # _log.debug("VPT: Erasing %s of %s, adding %s of %s",
        # key, ctx[:-1], newkey, ctx[1:])

    def searchforvpt(self, ctx):
        d = ctx.data
        predictor = self.pred(*self.args, **self.kwargs)
        while d.size > 0:
            k = hash(_wrapper(d))
            p = self.vpt.get(k, False)
            if p:
                predictor = p
                break
            d = d[:-1]
        return predictor

    @staticmethod
    @abstractmethod
    def getctx(ispace, ctx):
        """
        Choose specific context from information space.
        """
        return None

    @abstractmethod
    def updatecrit(self, *args, **kwargs):
        """
        Update criteria for best prediction.
        """
        return None

    @abstractmethod
    def choose_best(self, *args, **kwargs):
        """
        From all the predictions given by predefined context space
        choose the best one.
        """
        return None

    def step(self, truth, ispace, ctx):
        """
        Main execution cycle of Predictors using Information Space.

        This is the main execution cycle:
          1. Choose relevant Information Context
          2. For each element in context give a prediction
          3. Choose from each prediction the best one ('best' criteria to
          be defined by `choose_best` method)
          4. Update each predictor used for each element in the context
          5. Update the criteria for selection of best prediction
        """
        infocontext = self.getctx(ispace, ctx)
        predsdict = {self.predict(ctx): ctx for ctx in infocontext}
        best = self.choose_best(predsdict)
        if best == 'default':
            best = self.default.predict()
        _log.debug("Obj: %s with IC(%s): %s", self.obj,
                   ctx, len(infocontext.context))
        _log.info("Obj: %s by %s(%s) with options %s: %s",
                  self.obj, self, self.predictor, sorted(list(predsdict.keys())), best)
        _ = [self.update(truth, ctx) for ctx in infocontext]
        _ = self.updatecrit(truth, predsdict)
        self.obj += 1
        return best

    def __repr__(self):
        return str(self.name)


class Aggregate:

    def __init__(self, spacefeeder, manager, ctx, **kwargs):
        self.feeder = spacefeeder
        self.manager = manager
        self.ctx = ctx
        self.kwargs = kwargs

    def feed(self, seq, pa=True):
        self.manager.reset()
        tmp = self.feeder.feed(seq, **self.kwargs)
        predictions = [self.manager.step(t, space, self.ctx)
                       for t, space in tmp]
        if not pa:
            result = np.array(predictions)
        else:
            result = np.zeros(seq.shape).astype(seq.dtype)
            for i, v in enumerate(seq.sequence):
                result.flat[v] = predictions[i]
            result = PredictionArray(result)
        name = str(self.manager.predictor)
        self.manager.reset()
        return name, result
