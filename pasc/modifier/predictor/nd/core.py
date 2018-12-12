#!/usr/bin/env python
# coding: utf-8
"""
Multidimensional Predictors.
"""
from functools import namedtuple
from pasc.backend import NDPredictor
from pasc.modifier.predictor.nd import pascalnd as pnd  # import estimate
from pasc.objects.informationcontext import InformationContext as IC
import numpy as np

# TODO: Weights for dimensions are missing
# TODO: Acceptance of several NAN objects is missing
Prediction = namedtuple("Prediction", "pred, ctx")


class Pascal(NDPredictor):

    name = "Pascal ND (no dim preference)"

    def __init__(self, window, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.w = window

    def update(self, truth, ispace):
        pass

    def predict(self, ispace):
        preds = list()
        for ic in [x for x in ispace.space.keys() if x != 0]:
            for ctx in ispace[ic].context:
                p, ctx = self.estimate(ctx, self.w)
                preds.append(Prediction(p, ctx))
        best = self.choose(preds)
        return best

    def choose(self, preds):
        average = np.average([x.pred for x in preds]).astype(
            preds[0].ctx.data.dtype)
        return average

    def estimate(self, ctx, winsize):
        nctx = self.window(ctx, winsize)
        est = pnd.estimate(nctx.data)
        nctx.info['ndim'] = nctx.data.ndim
        return est.astype(ctx.data.dtype), nctx

    @staticmethod
    def window(ic, winsize=None):
        if winsize is None:
            ic.info['slices'] = slice(*winsize)
            return ic
        elif isinstance(winsize, int):
            winsize = tuple([winsize] * ic.data.ndim)
        assert len(winsize) == ic.data.ndim, "Winsize has not same shape"
        nanpos = pnd._get_nan_position(ic.data)
        shape = ic.data.shape
        slices = []
        for i in range(ic.data.ndim):
            if winsize[i] is None:
                mini, maxi = None, None
            else:
                mini = max(0, nanpos[i] - winsize[i])
                maxi = min(shape[i], nanpos[i] + winsize[i])
            slices.append(slice(mini, maxi))
        newinfo = ic.info.copy()
        newinfo['slices'] = slices
        return IC.create(ic.data[slices], **newinfo)
