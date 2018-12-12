#!/usr/bin/env python
# coding: utf-8
"""Subtractor modifier."""

from pasc.backend import BaseSubstractor  # Super
from pasc.objects.predictionarray import PredictionArray as PA  # Input
from pasc.objects.integerarray import IntegerArray as IA  # Input
from pasc.objects.residualarray import ResidualArray as RA  # Output
import numpy as np


class XOR(BaseSubstractor):

    name = "XOR Subtractor"

    @staticmethod
    def subtract(predictionarray, integerarray):
        predictionarray, integerarray = _check_input(
            predictionarray, integerarray)
        result = np.bitwise_xor(predictionarray.array, integerarray.array)
        return RA(result)


class FPD(BaseSubstractor):

    name = "FP difference"

    @staticmethod
    def subtract(predictionarray, integerarray):
        predictionarray, integerarray = _check_input(
            predictionarray, integerarray)
        result = np.subtract(predictionarray.array, integerarray.array)
        result = np.absolute(result)
        return RA(result)


def _check_input(predictionarray, integerarray):
    if not isinstance(predictionarray, PA):
        err = "Expected PredictionArray, got {}".format(
            type(predictionarray))
        raise TypeError(err)
    if not isinstance(integerarray, IA):
        err = "Expected IntegerArray, got {}".format(type(integerarray))
        raise TypeError(err)
    return predictionarray, integerarray
