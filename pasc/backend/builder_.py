#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interface for Builder modifiers."""

from abc import ABCMeta, abstractmethod, abstractproperty
from pasc.toolbox import check_methods


class BuilderInterface(metaclass=ABCMeta):
    """Define Information Space for each value.
    """

    __slots__ = ()

    @abstractmethod
    def build_informationspace(self):
        """Build input for predictors."""
        return []

    @classmethod
    def __subclasshook__(cls, C):
        if cls is BuilderInterface:
            return check_methods(C, "build_informationspace")
        return NotImplemented


class BaseBuilder(BuilderInterface):

    @abstractproperty
    def name(self):
        return NotImplemented

    def __repr__(self):
        return str(self.name)
