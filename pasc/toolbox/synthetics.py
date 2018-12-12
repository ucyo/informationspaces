#!/usr/bin/env python
# coding: utf-8
"""
Generating artificial test files for prediction.
"""

import logging
from itertools import islice
from functools import wraps
import numpy as np

def logseed(func):
    LOGG = logging.getLogger()

    @wraps(func)
    def wrapper(*args, **kwargs):
        seed = np.random.randint(2**32)
        seed = kwargs.pop('seed', seed)
        np.random.seed(seed)
        LOGG.warning("Calling '%s()' with args=%s, kwargs=%s, seed=%s",
                     func.__name__, args, kwargs, seed)
        return func(*args, **kwargs)
    return wrapper


class Sequence:

    @staticmethod
    @logseed
    def const(num=None):
        num = np.random.randint(1, 100) if num is None else num
        while True:
            yield num

    @staticmethod
    @logseed
    def step(step):
        val = 0
        while True:
            val += step
            yield val

    @staticmethod
    @logseed
    def pattern(N, args=None, shuffle=False):
        nums = np.arange(N) if args is None else args
        if shuffle:
            nums = np.random.shuffle(nums)
        while True:
            for x in nums:
                yield x

    @staticmethod
    @logseed
    def step_probability(step, probability=.8):
        val = 0
        while True:
            growth = np.random.random_sample()
            if growth <= probability:
                val += step
            yield val

    @staticmethod
    @logseed
    def gauss(mu, sigma):
        while True:
            yield np.random.normal(loc=mu, scale=sigma)

    @staticmethod
    @logseed
    def poisson(lam):
        while True:
            yield np.random.poisson(lam)

    @staticmethod
    @logseed
    def exponential(scale):
        while True:
            yield np.random.exponential(scale)

    @staticmethod
    @logseed
    def laplace(mu, sigma):
        while True:
            yield np.random.laplace(loc=mu, scale=sigma)


class MultiDim:

    @staticmethod
    def createNd(shape, method, *args, **kwargs):
        shape = shape[::-1]
        if len(shape) == 1:
            func = getattr(Sequence, method, False)
            if not func:
                raise "Wrong method"
            size = shape[0]
            return np.stack([x for x in islice(func(*args, **kwargs), size)], axis=0)
        return np.stack([MultiDim.createNd(shape[:-1], method, *args, **kwargs)
                         for i in range(shape[-1])], axis=len(shape) - 1).T


if __name__ == '__main__':
    limit = 10
    generators = (
        Sequence.const(None),
        Sequence.step(42),
        Sequence.step_probability(42, probability=.4),
        Sequence.pattern(8),
        Sequence.gauss(1, 1),
        Sequence.poisson(4),
        Sequence.exponential(3),
        Sequence.laplace(1, 1),
        MultiDim.createNd((3, 2), 'const', None),
        MultiDim.createNd((3, 2), 'gauss', 1, 10),
    )
    for patterns in generators:
        result = list(islice(patterns, limit))
        print(result)
