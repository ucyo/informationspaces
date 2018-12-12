#!/usr/bin/env python
# coding: utf-8
"""Tests for graphtheory."""

from pasc.toolbox import graphtheory as gt
import pandas as pd
import pytest

ADJMATRICES = [
    #    A
    #  B   E
    # C D
    [[0, 1, 0, 0, 1],
     [1, 0, 1, 1, 0],
     [0, 1, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [1, 0, 0, 0, 0]],
]

RESULTS_BFS = [
    ['A', 'E', 'B', 'D', 'C'],
]

RESULTS_DFS = [
    ['A', 'B', 'C', 'D', 'E'],
]

BFS = zip(ADJMATRICES, RESULTS_BFS)
DFS = zip(ADJMATRICES, RESULTS_DFS)


@pytest.mark.parametrize('adjmatrix,expected', BFS)
def test_weighted_bfs(adjmatrix, expected):
    namedmatrix = gt.nameAdjMatrix(adjmatrix)
    _, result = gt.search.BFSmatrix(adjmatrix=namedmatrix, startnode='A')
    assert list(result) == expected


@pytest.mark.parametrize('adjmatrix,expected', DFS)
def test_weighted_dfs(adjmatrix, expected):
    namedmatrix = gt.nameAdjMatrix(adjmatrix)
    _, result = gt.search.DFS(adjmatrix=namedmatrix, startnode='A')
    assert list(result) == expected


VALID_ADJMATRIX = [
    [[0, 0, 0, 1],
     [0, 0, 1, 1],
     [0, 1, 0, 1],
     [1, 1, 1, 0]],
]

@pytest.mark.parametrize('adjmatrix', VALID_ADJMATRIX)
def test_search_valid_adjmatrix(adjmatrix):
    named = gt.nameAdjMatrix(adjmatrix, 'a')
    _ = gt.search.BFSmatrix(adjmatrix=named, startnode='A')
    assert True



INVALID_ADJMATRIX = [
    # not symmetric
    [[0, 1, 0, 1],
     [0, 0, 1, 1],
     [0, 1, 0, 1],
     [1, 1, 1, 0]],
]


@pytest.mark.parametrize('adjmatrix', INVALID_ADJMATRIX)
def test_invalid_search_adjmatrix(adjmatrix):
    named = gt.nameAdjMatrix(adjmatrix, 'a')
    with pytest.raises(AssertionError):
        _ = gt.search.BFSmatrix(adjmatrix=named, startnode='A')
        _ = gt.search.BFSmatrix(adjmatrix=adjmatrix, startnode='A')




@pytest.mark.parametrize('adjmatrix', ADJMATRICES)
def test_naming_of_adjmatrix(adjmatrix):
    namedmatrix = gt.nameAdjMatrix(adjmatrix)
    assert isinstance(namedmatrix, pd.DataFrame)
    assert all(namedmatrix.columns == namedmatrix.index)
    assert namedmatrix.index.size == pd.np.sqrt(namedmatrix.size)


RANDOM = [
    (5, (0, 3)),
    (13, (0, 1)),
]


@pytest.mark.parametrize('N,limits', RANDOM)
def test_random_adjmatrix_generator(N, limits):
    mi, ma = min(limits), max(limits)
    adjmatrix = gt.random.adjMatrixGenerator(N=N, limits=limits)
    assert all([adjmatrix[x, y] == adjmatrix[y, x]
                for x in range(N) for y in range(N)])
    assert adjmatrix.shape == (N, N)
    assert adjmatrix.max() <= ma
    assert adjmatrix.min() >= mi


@pytest.mark.parametrize('execution_number', range(20))
def test_random_input(execution_number):
    r = pd.np.random.randint(1, 633)
    random = gt.random.namedAdjMatrixGenerator(N=5, seed=r, mode='n')
    _, result = gt.search.BFSmatrix(adjmatrix=random, startnode='1')
    assert len(result) == pd.np.sqrt(random.size)
    assert len(result) == len(set(result))


NOT_FULLY_CONNECTED_GRAPH = [
    # C not connected
    [[0, 1, 0, 0, 1],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1],
     [1, 1, 0, 1, 0]],
    # C and D not connected
    [[0, 1, 0, 0, 1],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0]],
    # C and D connected to each other
    [[0, 1, 0, 0, 1],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0],
     [1, 1, 0, 0, 0]],
]


@pytest.mark.parametrize('adjmatrix', NOT_FULLY_CONNECTED_GRAPH)
def test_not_fully_connected_graph(adjmatrix):
    named = gt.nameAdjMatrix(adjmatrix, mode='n')
    _, result = gt.search.BFSmatrix(adjmatrix=named, startnode='1')
    assert len(result) == pd.np.sqrt(named.size)
