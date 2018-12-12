#!/usr/bin/env python
# coding: utf-8
"""Adjacent matrices and lists."""

from pasc.toolbox.graphtheory.elements.cells import fromIdx, Cell
import numpy as np
import pandas as pd


def generateAdjListAtDistance(distance, shape):
    """Generate adjacent list from distance pattern.

    Generates a adjacent list with neighbours at certain distance.
    The shape argument is needed to eliminate non-existing neighbours
    at array borders.

    Arguments
    =========
    distance : int
        Distance from source cells and neighbour cells to be considered.
    shape : tuple(int)
        Shape of original array.

    Returns
    =======
    adjList : dict(cell:list(cell))
        Adjacent list for each cell. With `cell.val` representing the name
        of the node.
    """
    if isinstance(distance, (list, tuple)):
        return generateAdjListAtDistanceLIST(distance, shape)

    tmpdata = np.arange(np.prod(shape))
    # nodenames = (tmpdata + 1).astype('<U3')
    nodenames = np.array([str(x+1) for x in tmpdata]).reshape(shape)

    adjList = dict()
    for cellAtIdx in range(nodenames.size):
        cell = fromIdx(idx=cellAtIdx, shape=shape,
                       value=nodenames.flat[cellAtIdx])
        neighbours = [x.index(shape)
                      for x in cell.neighboursAtDistance(distance, shape)]
        neighbours = [fromIdx(idx=int(x), shape=shape,
                              value=nodenames.flat[int(x)]) for x in neighbours]
        adjList[cell] = neighbours
    return adjList


def generateAdjListAtDistanceLIST(distances, shape):
    """Merge function if several distances are taken into account."""
    def mergedict(dict1, dict2):
        result = dict()
        keys = set(dict1.keys()).union(dict2.keys())
        for k in keys:
            result[k] = set(dict1.get(k, [])).union(dict2.get(k, []))
        return result

    result = dict()
    while distances:
        tmp = generateAdjListAtDistance(distances.pop(), shape)
        result = mergedict(result, tmp)
    return result


def addweights(adjlist, weights, default):
    """Add weights to adjacent list to prioritize certain neighbours.

    The weights added by this method prioritizes (or not) the neighbours
    on a siblings level.

    Arguments
    =========
    adjList : dict(cell:list(cell))
        Adjacent list for each cell. With `cell.val` representing the name
        of the node.
    weights : dict(direction:int)
        Weights for each direction.
    default : int
        Default weight if direction is not in weightlist.

    Returns
    =======
    final : dict(cell:dict(cell:int))
        Weighted adjacent list.
    """
    if not default > 0:
        raise BaseException('default <= 0')
    result = {k: {(x - k): x.val for x in v} for k, v in adjlist.items()}
    final = dict()
    for k, v in result.items():
        neighbours = {}
        for direction, val in v.items():
            cell = direction + k
            cell = Cell(cell.coord, val)
            neighbours[cell] = weights.get(direction, default)
        final[k] = neighbours
    return final


def wadjlist2adjmatrix(wadjlist, shape):
    """Transform weighted adjacent list into adjacent matrix.

    Arguments
    =========
    wadjlist : dict(cell:dict(cell:int))
        Weighted adjacent list.
    shape : tuple(int)
        Shape of original array.
    Returns
    =======
    matrix : pd.DataFrame
        Adjacent Matrix.
    """
    matrix = emptyAdjMatrix(np.prod(shape))
    for sourcecell, neighbours in wadjlist.items():
        for neighbour, neighbourweight in neighbours.items():
            matrix.loc[sourcecell, neighbour] = neighbourweight
    return matrix


def emptyAdjMatrix(N, mode='n'):
    empty = np.zeros((N, N), dtype=int)
    return nameAdjMatrix(empty, mode=mode)


def nameAdjMatrix(adjmatrix, mode=None):
    """Name nodes of adjacent matrix either (n)umerical or (a)lpanumerical."""
    adjmatrix = np.array(adjmatrix)
    N = adjmatrix.shape[0]

    if not mode and N <= 26:
        mode = 'a'
    elif not mode:
        mode = 'n'
    elif mode not in ('a', 'n'):
        raise Exception('No valid mode')
    alphabeticnames = [chr(x + 64) for x in range(1, N + 1)]
    numericalnames = [str(x) for x in range(1, N + 1)]
    names = alphabeticnames if mode == 'a' else numericalnames
    df = pd.DataFrame(data=adjmatrix, index=names, columns=names)
    return df


def nodelist2idxlist(nodelist):
    if isinstance(nodelist, np.ndarray):
        result = nodelist.astype(int) - 1
    else:
        result = np.array(nodelist).astype(int) - 1
    return result
