#!/usr/bin/env python
# coding: utf-8
"""Interface for Encoder modifiers."""

from abc import ABCMeta, abstractmethod
from pasc.toolbox import check_methods

class EncoderInterface(metaclass=ABCMeta):
    """Modifier to encode residual array."""

    __slots__ = ()

    @abstractmethod
    def encode(self):
        """Encode residual array."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is EncoderInterface:
            return check_methods(C, "encode")
        return NotImplemented
