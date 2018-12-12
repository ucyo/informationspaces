#!/usr/bin/env python
# coding: utf-8
"""Cell + Direction representation of an element."""

from itertools import product
from pasc.toolbox import check_methods
from pasc.toolbox.graphtheory.elements import BaseCell, BaseDirection
import numpy as np


class Direction(BaseDirection):
    """Movement direction from a cell to another."""

    def __init__(self, direction):
        self._direction = np.array(direction)

    @property
    def direction(self):
        return self._direction

    def __hash__(self):
        return hash(tuple(self.direction))

    def __eq__(self, other):
        if not isinstance(other, Direction):
            raise NotImplementedError
        else:
            return all(self.direction == other.direction)

    def __repr__(self):
        obj = ",".join([str(x) for x in self.direction])
        return "Direction[{}]".format(obj)

    def __mul__(self, factor):
        return Direction(self.direction * factor)

    def __add__(self, other):
        if isinstance(other, Cell):
            return other.__add__(self)

    def __abs__(self):
        return Direction(abs(self.direction))

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Direction:
            return check_methods(C, "direction")
        return NotImplemented


class Cell(BaseCell):
    """Cell object."""

    def __init__(self, coord, val):
        self._coord = np.array(coord)
        self._val = val

    @property
    def val(self):
        return self._val

    @property
    def coord(self):
        return self._coord

    def index(self, shape):
        return np.ravel_multi_index(self._coord, shape)

    @property
    def relneighbours(self):
        """Relative neighbours of the Cell.

        Note
        ====
        Relative means the steps in each dimension to be taken to arrive
        at a neighbouring Cell. At the moment these are the immediate
        neighbours.

        Returns
        =======
        result : list of Directions
            Directions to move to get to next neighbours.
        """
        dims = len(self.coord)
        ranges = [x * y for x, y in product(range(1, 2), [-1, 1])] + [0]
        combinations = [Direction(x) for x in list(
            product(ranges, repeat=dims)) if self + Direction(x)]
        return combinations

    @property
    def neighbours(self):
        """Neighbours of Cell.

        Note
        ====
        Actual neighbours given by coordinates (instead of relative
        distance contrary to `relneighbours`).

        Returns
        =======
        result : list of Cells
            Neighbouring Cells of current Cell.
        """
        return [self + x for x in self.relneighbours]

    @staticmethod
    def pathdistance(relcoord):
        """Distance to relative coordinate.

        Arguments
        =========
        relcoord : Direction
            Relative coordinate of Cell to calculate distance.

        Returns
        =======
        result : int
            Distance to Cell with direction of `relcoord`.
        """
        return sum([x for x in abs(np.array(relcoord)).direction])

    def neighboursAtDistance(self, distance, shape):
        """All neighbouring Cells with certain distance.

        Arguments
        =========
        distance : int
            Distance length.

        Returns
        =======
        disdict : list of Cells
            All Cells in the neighbourhood with certain distance
        """
        distances = np.array([self.pathdistance(x)
                              for x in self.relneighbours])
        disdict = [self.neighbours[i]
                   for i, x in enumerate(distances) if x == distance]
        result = list()
        for candidate in disdict:
            try:
                _ = candidate.index(shape=shape)
            except ValueError:
                pass
            else:
                result.append(candidate)
        return result

    @property
    def distance(self, shape):
        """All neighbouring points and their distances.

        Returns
        =======
        result : dict
            All neighbouring points with distances as keys and list of Cells
            as values.
        """
        return {y: self.neighboursAtDistance(y, shape)
                for y in range(len(self.coord) + 1)}

    def __add__(self, other):
        if isinstance(other, Direction):
            newcoord = self.coord + other.direction
            if not all(newcoord >= 0):
                #  print("Addition dropped.")
                return None
            return Cell(newcoord, self.val)
        else:
            raise NotImplementedError("Nope")

    def __sub__(self, other):
        if isinstance(other, Direction):
            return self.__add__(other * -1)
        elif isinstance(other, Cell):
            return Direction([x - y for x, y in zip(self.coord, other.coord)])
        else:
            message = "Nope: {},{}".format(type(self), type(other))
            raise NotImplementedError(message)

    def __hash__(self):
        return hash(tuple(self.coord))  # + hash(self.val)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        obj = ",".join([str(x) for x in self.coord])
        return "Cell[{}]".format(obj)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Cell:
            return check_methods(C, "coord", "val", "neighbours")
        return NotImplemented


def fromIdx(idx, shape, value):
    """Create a Cell object from index position (and shape).

    Arguments
    =========
    idx : int
        Index position of Cell.
    shape : tuple
        Shape of origin array.
    value : numerical
        Value of the Cell content.

    Returns
    =======
    result : Cell
        A Cell object with coordinates at given index position.
    """
    assert isinstance(idx, int)
    coords = np.unravel_index(indices=idx, dims=shape)
    return Cell(coords, value)
