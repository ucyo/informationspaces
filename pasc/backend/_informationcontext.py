#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interface for Information Context objects."""

from abc import ABCMeta, abstractproperty
from pasc.toolbox import check_methods


class InformationContextInterface(metaclass=ABCMeta):
    """InformationContext for a single point in certain dimension profiles.

    Each np.array has sinlge np.nan object which needs to be predicted.

    Attributes
    ==========
    context : list(np.ndarray)
        List of context elements with same number of dimensions.
    """

    __slots__ = ()

    @abstractproperty
    def context(self):
        """All contexts for single point."""
        return list()

    @abstractproperty
    def dims(self):
        """Number of dimensions per context."""
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if cls is InformationContextInterface:
            return check_methods(C, "context", "dims")
        return NotImplemented


class BaseInformationContext(InformationContextInterface):

    def __repr__(self):
        return "IC{}: {}".format(self.dims, self.context)

    def __getitem__(self, name):
        return self.context[name]
