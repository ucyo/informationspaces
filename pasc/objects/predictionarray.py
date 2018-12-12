#!/usr/bin/env python
# coding: utf-8
"""Classes of Prediction Arrays."""

import numpy as np
from pasc.backend import BaseArray


class PredictionArray(BaseArray):
    """Basic Prediction Array format.
    """

    def __init__(self, arr):
        self.array = arr

    @property
    def valid_dtypes(self):
        return (np.int32, np.int64, int, np.uint32, np.uint64)
