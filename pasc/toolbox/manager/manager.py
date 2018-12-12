#!/usr/bin/env python
# coding: utf-8
"""
Manager building on BaseManager
"""
import logging
from pasc.toolbox.manager import BaseManager
from pasc.objects.informationcontext import InformationContext
from pasc.toolbox.flood import getNAN
import numpy as np
_log = logging.getLogger(__name__)


class AverageManager(BaseManager):
    """
    Context Manager with no preferences between the contexts of
    an Information Context an chooses the average.
    """

    name = "Average Manager"

    def updatecrit(self, truth, opreds):
        # truth {truth value}
        # opreds {key: prediction, value: context}
        pass

    @staticmethod
    def getctx(ispace, val):
        icontext = ispace[val]
        try:
            INTNAN = getNAN(icontext.context[0].data.dtype)
        except IndexError:
            return icontext
        icontext = [x if x.data[0] != INTNAN else InformationContext.create(
            data=x.data[::-1], **x.info) for x in icontext.context]
        return InformationContext(icontext)

    def choose_best(self, opreds):
        # opreds {key: prediction, value: context}
        preds = [x for x in opreds.keys() if x != 0]
        result = np.average(preds).astype(np.int32) if preds else "default"
        return result


class ReproduceManager(AverageManager):

    name = "Reproduce Manager"

    def choose_best(self, opreds):
        predict = "default"
        for k, c in opreds.items():
            if c.info['id'] == 2 and k != 0:
                predict = k
                break
            elif c.info['id'] != 2:
                predict = k
        _log.debug("OPreds: %s, choosen %s", opreds, predict)
        return predict


class LastBestManager(AverageManager):

    name = "Last Best Manager"

    def updatecrit(self, truth, opreds):
        preds = {k: v for k, v in opreds.items() if k != 0}
        if not preds:
            self._lastbestID = None
        else:
            result = sorted(preds.items(), key=lambda x: abs(x[0] - truth))
            self._lastbestID = result[0][1].info['id']
            _log.debug("Last Best ID - Opreds: %s - Truth: %s - LastBest: %s",
                       opreds, truth, self._lastbestID)

    def choose_best(self, opreds):
        predict = "default"
        try:
            for k, c in opreds.items():
                if c.info['id'] == self._lastbestID and k != 0:
                    predict = k
                    break
                elif c.info['id'] != self._lastbestID:
                    predict = k
        except AttributeError:
            pass
        _log.debug("OPreds: %s, choosen %s", opreds, predict)
        return predict


class MinManager(AverageManager):
    """
    Context Manager which takes the smallest prediction.
    """

    name = "Min Manager"

    def choose_best(self, opreds):
        preds = sorted([x for x in opreds.keys() if x != 0])
        if not preds:
            result = "default"
        else:
            result = preds[0]
        return result


class MaxManager(AverageManager):
    """
    Context Manager which takes the biggest prediction.
    """

    name = "Max Manager"

    def choose_best(self, opreds):
        preds = sorted([x for x in opreds.keys() if x != 0])
        if not preds:
            result = "default"
        else:
            result = preds[-1]
        return result


class CountManager(AverageManager):
    """
    Manager using context with biggest number of points.
    """

    name = "Count Manager"

    def choose_best(self, opreds):
        if not opreds:
            return "default"
        preds = {k: v for k, v in opreds.items() if k != 0}
        result = sorted(preds.items(), key=lambda x: x[1].size, reverse=True)
        try:
            return result[0][0]
        except IndexError:
            return "default"
