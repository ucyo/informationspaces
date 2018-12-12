#!/usr/bin/env python
# coding: utf-8
"""Interface for Writer modifiers."""

from abc import ABCMeta, abstractmethod
from pasc.toolbox import check_methods

class WriterInterface(metaclass=ABCMeta):
    """Modifier to write file on disk in pasc format."""

    __slots__ = ()

    @abstractmethod
    def write(self):
        """Write file on disk in pasc format."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is WriterInterface:
            return check_methods(C, "write")
        return NotImplemented
