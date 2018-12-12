#!/usr/bin/env python
# coding: utf-8
"""Random adjacent matrix generator."""

from pasc.toolbox.graphtheory.adjacency import nameAdjMatrix
import numpy as np


def adjMatrixGenerator(N=None, seed=42, limits=(0, 1)):
    """Generate 'random' adjacent matrix of size NxN and appropiate weightlimits.

    Note
    ====
    Reproducable via 'seed' argument.

    Arguments
    =========
    N : int
        Size of the adjacent matrix. This represents the number of nodes.
    seed: int
        Seed value for reproduction of random values.
    weightlimits: tuple(int)
        Min and max of weights for edges in the graph.

    Returns
    =======
    m : np.ndarray
        An adjacent matrix of a random Graph.
    """
    np.random.seed(seed)
    if not N:
        N = np.random.randint(low=2, high=20)
    mi, ma = min(limits), max(limits) + 1

    a = np.random.randint(mi, ma, size=(N, N))
    m = np.tril(a) + np.tril(a, -1).T
    m[np.diag_indices_from(m)] = 0
    return m


def namedAdjMatrixGenerator(N, seed=42, limits=(0, 1), mode='a'):
    """Generate a named random adjacent matrix."""
    adjmatrix = adjMatrixGenerator(N=N, seed=seed, limits=limits)
    namedadjmatrix = nameAdjMatrix(adjmatrix=adjmatrix, mode=mode)
    return namedadjmatrix
