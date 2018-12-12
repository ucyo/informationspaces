#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interface for Information Space objects."""

from abc import ABCMeta, abstractproperty
from pasc.toolbox import check_methods


class InformationSpaceInterface(metaclass=ABCMeta):
    """InformationSpace with different information contexts for
    a single point.

    Attributes
    ==========
    space : dict(InformationContexts)
        A dictionary of Information Context elements
    """

    __slots__ = ()

    @abstractproperty
    def space(self):
        """All contexts for single point."""
        return dict()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is InformationSpaceInterface:
            return check_methods(C, "space")
        return NotImplemented


class BaseInformationSpace(InformationSpaceInterface):

    def __repr__(self):
        return "Space {}".format(self.space)

    def __getitem__(self, name):
        return self.space[name]
