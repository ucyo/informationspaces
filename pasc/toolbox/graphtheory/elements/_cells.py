#!/usr/bin/env python
# coding: utf-8
"""Interfaces for Direction and Cell objects.

The Direction and Cell objects are tightly coupled. A Cell object can move to
another Cell object only with the help von a Direction object.
"""

from abc import ABCMeta, abstractproperty
from pasc.toolbox import check_methods


class BaseDirection(metaclass=ABCMeta):
    """Direction object defining movement from Cell to Cell.

    Attributes
    ==========
    direction : tuple
        Movment direction from one Cell to the other.
    """

    __slots__ = ()

    @abstractproperty
    def direction(self):
        """Direction in which to move."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is BaseDirection:
            return check_methods(C, "direction")
        return NotImplemented


class BaseCell(metaclass=ABCMeta):
    """Cell object.

    Attributes
    ==========
    coord : np.array
        Coordinates of the Cell in the original array.
    val : numeric
        Value of the Cell in the original array.
    neighbours: List
        Neighbouring Cells of this Cell.
    """

    __slots__ = ()

    @abstractproperty
    def coord(self):
        """Coordinates of the Cell."""
        return None

    @abstractproperty
    def val(self):
        """Value of Cell."""
        return None

    @abstractproperty
    def neighbours(self):
        """List of neighbouring Cells."""
        return []

    @classmethod
    def __subclasshook__(cls, C):
        if cls is BaseCell:
            return check_methods(C, "coord", "val", "neighbours")
        return NotImplemented
