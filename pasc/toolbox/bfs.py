#!/usr/bin/env python
# coding: utf-8
"""Subbenchmarking."""

from collections import deque
from numbers import Number
from itertools import product, chain
from functools import lru_cache, partial
import numpy as np


@lru_cache(maxsize=128)
def all_neighbours(dims):
    """All relative coordinates of adjacent cells."""
    return [x for x in list(product([-1, 1, 0], repeat=dims))]


def neighbour_idx(idx, distance, shape, weights, default=1):
    """Get adjacent idx at certain distance with proper weights."""
    coord = np.unravel_index(idx, shape)
    dims = len(shape)
    neighbours = tuple(all_neighbours(dims))

    dis_neighbours = neighbours_at(distance, neighbours, weights, default)
    idx_neighbours = dict()
    for k, n in dis_neighbours:
        try:
            idx_neighbours[np.ravel_multi_index(
                np.array(k) + coord, shape)] = n
        except ValueError:
            pass
    return idx_neighbours


def pack_weights(weights):
    """Pack weights from dict to tuple to make them hashable."""
    return tuple((k, v) for k, v in weights.items())


def unpack_weights(weights):
    """Unpack weights from tuple to dict to work with them."""
    return {k: v for k, v in weights}


@lru_cache(128)
def neighbours_at(distance, neighbours, weights, default=1):
    """Indices of adjacent cells at certain distance."""
    weights = unpack_weights(weights)
    weighted_neighbours = {x: weights.get(
        x, default) for x in neighbours if np.sum(np.abs(x)) in distance}
    result = sorted(weighted_neighbours.items(),
                    key=lambda x: x[1], reverse=True)
    return result


@lru_cache(128)
def BFSms(shape, startidx, distance, weights, default=1, marked=None):
    """A faster implementation of BFS method using properties of the cube.
    """
    if isinstance(distance, Number):
        distance = (distance,)
    maxsize = int(np.prod(shape))
    search = deque(maxlen=maxsize)
    if marked is None:
        marked = deque(maxlen=maxsize)
    else:
        marked = deque(marked, maxlen=maxsize)

    search.append(startidx)
    while search:
        element = search.popleft()
        if element in marked:
            continue
        neighbours = neighbour_idx(
            idx=element, distance=distance, shape=shape, weights=weights, default=default)
        for k in neighbours:
            search.append(k)
        marked.append(element)
    while len(marked) != maxsize:
        idx = startidx
        while idx in marked:
            idx += 1
        _, marked = BFSms(shape, idx, distance, weights, default, tuple(marked))
    return startidx, marked


BFSCheq = partial(BFSms, distance=2,)
BFSBlos = partial(BFSms, distance=1,)
BFSBloc = partial(BFSms, distance=(1, 2, 3))


# If there are no weights set one can use classical BFS


class BFSSetup:
    """Set up of BFS algorithm using distance and shape.
    """

    def __init__(self, shape, distance):
        self.distance = distance
        self.shape = shape
        self.default = 1
        self.weights = ()

    def __getitem__(self, name):
        return neighbour_idx(idx=name, distance=self.distance,
                             shape=self.shape, weights=self.weights,
                             default=self.default)
_CheqSetup = partial(BFSSetup, distance=(2,))
_BlosSetup = partial(BFSSetup, distance=(1,))
_BlocSetup = partial(BFSSetup, distance=(1, 2, 3,))


def classicBFS(g, start):
    """BFS with no weights."""
    queue, enqueued = deque([(None, start)]), set([start])
    while queue:
        _, n = queue.popleft()
        yield n
        new = set(g[n]) - enqueued
        enqueued |= new
        queue.extend([(n, child) for child in new])


def CheqNoWeights(shape, startidx):
    """Chequerboard BFS without weights."""
    gen1 = classicBFS(_CheqSetup(shape), startidx)
    gen2 = classicBFS(_CheqSetup(shape), startidx+1)
    return chain(gen1, gen2)

def BlosNoWeights(shape, startidx):
    """Blossom BFS without weights."""
    return classicBFS(_BlosSetup(shape), startidx)

def BlocNoWeights(shape, startidx):
    """Block BFS without weights."""
    return classicBFS(_BlocSetup(shape), startidx)
