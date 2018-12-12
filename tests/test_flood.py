#!/usr/bin/env python
# coding: utf-8
"""Tests for flood module."""

from pasc.toolbox import flood as fl
import pytest


def test_NoneT():
    assert str(fl.NoneT()) == 'All'
