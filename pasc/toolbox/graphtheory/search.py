#!/usr/bin/env python
# coding: utf-8
"""Search algorithms for graphtheory."""

import math
import logging
from collections import deque
import pandas as pd
from pasc.toolbox import setupLog

setupLog()
LOG = logging.getLogger(__name__)


def sorted_child_nodes(adjmatrix, node, lowest_first=False):
    """Sort all child notes accordingly of adjacent matrix."""
    row = adjmatrix[node][adjmatrix[node] > 0]
    result = row.sort_values(ascending=lowest_first).index
    return result


def drop(adjmatrix, node):
    tmp = adjmatrix.drop(node, 0)
    dropped = tmp.drop(node, 1)
    return dropped


def dropAll(adjmatrix, exceptionlist=None):
    if not exceptionlist:
        exceptionlist = []
    tobedropped = [x for x in adjmatrix.columns if x not in exceptionlist]
    while tobedropped:
        node = tobedropped.pop()
        adjmatrix = drop(adjmatrix, node)
    return adjmatrix


def BFSlist(adjlist, startnode):
    """Breadth First Search based on adjacent list."""
    search = deque([startnode], maxlen=len(adjlist))
    marked = deque([], maxlen=len(adjlist))

    while search:
        element = search.popleft()
        if element in marked:
            continue
        children = sorted(adjlist[element])
        for child in children:
            if child not in search:
                search.append(child)
        marked.append(element)
    if len(marked) != len(adjlist):
        marked = check_missing_elements_list(
            adjlist, marked, BFSlist, startnode)
    return startnode, marked


def check_missing_elements_list(adjlist, marked, method, oldstart=None):
    """Check missing elements in adjacent list."""
    for node in marked:
        for k in [k for k, v in adjlist.items() if node in v]:
            del adjlist[k][node]
        del adjlist[node]

    _, add = method(adjlist, list(adjlist)[0])
    marked.extend(add)
    return marked


def BFSmatrix(adjmatrix, startnode):
    """Breadth First Search based on adjacent matrix."""
    assert check_adjacent_matrix(adjmatrix)
    size = adjmatrix.index.size
    LOG.info("Logging matrix %s", adjmatrix)
    search = deque([startnode], maxlen=size)
    marked = deque([], maxlen=size)

    while search:
        element = search.popleft()
        if element in marked:
            continue
        children = sorted_child_nodes(adjmatrix, element)
        LOG.info("Children of %s are %s", element, children)
        for child in children:
            if child not in search:
                search.append(child)
        marked.append(element)

    marked = check_missing_elements_matrix(
        adjmatrix=adjmatrix, marked=marked, method=BFSmatrix, oldstart=startnode)
    return startnode, marked


def check_missing_elements_matrix(adjmatrix, marked, method, oldstart):
    """Check missing elements in adjacent matrix."""
    size = adjmatrix.columns.size

    # if adjmatrix was not fully connected
    if len(marked) != size:
        missingelements = [x for x in adjmatrix.columns if x not in marked]
        subAdjMatrix = dropAll(adjmatrix, missingelements)
        new_startnode = find_nearest(subAdjMatrix, oldstart)
        _, add = method(subAdjMatrix, new_startnode)
        marked.extend(add)
    return marked


def check_adjacent_matrix(adjmatrix):
    assert isinstance(adjmatrix, pd.DataFrame)
    # N = adjmatrix.columns.size
    assert len(set(adjmatrix.shape)) == 1
    # assert all([adjmatrix.values[x, y] == adjmatrix.values[y, x]
    # for x in range(N) for y in range(N)])
    return True


def DFS(adjmatrix, startnode):
    """Depth First Search based on adjacent matrix."""
    assert check_adjacent_matrix(adjmatrix)
    size = adjmatrix.index.size
    search = deque([startnode], maxlen=size)
    marked = deque([], maxlen=size)

    while search:
        element = search.pop()
        if element in marked:
            continue
        children = sorted_child_nodes(adjmatrix, element)
        LOG.info("Children of %s are %s", element, children)
        for child in children:
            if child not in search:
                search.append(child)
        marked.append(element)

    marked = check_missing_elements_matrix(
        adjmatrix=adjmatrix, marked=marked, method=DFS, oldstart=startnode)
    return startnode, marked


def find_nearest(array, value):
    array = array.columns.values.astype(int)
    value = int(value)

    idx = pd.np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        result = str(array[idx - 1])
    else:
        result = str(array[idx])
    return result
