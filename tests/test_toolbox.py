#!/usr/bin/env python
# coding: utf-8
""""Tests for toolbox."""

import pytest
from pasc import toolbox
import xarray as xr


CORRECT_KEYS = ['pre']
UNKONWN_KEYS = ['hi']


@pytest.mark.parametrize('key', CORRECT_KEYS)
def test_dataload(key):
    assert isinstance(toolbox.load_data(key), xr.Dataset)


@pytest.mark.parametrize('key', UNKONWN_KEYS)
def test_err_dataload(key):
    with pytest.raises(KeyError) as err:
        _ = isinstance(toolbox.load_data(key), xr.Dataset)
    assert "Unknown" in str(err)
