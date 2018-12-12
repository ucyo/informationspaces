#!/usr/bin/env python
# coding: utf-8
"""Types of Information Space."""

from pasc.backend import BaseInformationSpace, BaseInformationContext as bIC
from pasc.objects.informationcontext import InformationContext as IC


class InformationSpace(BaseInformationSpace):

    name = "InformationSpace"

    def __init__(self, dictionary):
        self.space = dictionary

    def _get_space(self):
        return self._space

    def _set_space(self, value):
        if not isinstance(value, dict):
            err = "Expected dict, got {}.".format(type(value))
            raise TypeError(err)
        additionalValidTypes = (list, tuple)
        if all([isinstance(v, additionalValidTypes) for _, v in value.items()]):
            asIC = dict()
            for k, v in value.items():
                asIC[k] = IC(v)
            value = asIC
        if not all([isinstance(v, bIC) for _, v in value.items()]):
            err = "Expected IC compatible datatype."
            raise TypeError(err)
        self._space = value
    space = property(_get_space, _set_space)

    def __repr__(self):
        return str({k:v for k,v in sorted(self.space.items())})
