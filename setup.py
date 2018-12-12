#!/usr/bin/env python
# coding: utf-8
"""Minimal setup."""

import os
import re
from setuptools import setup, find_packages

PROJECT = 'pasc'


def get_property(prop, project):
    """Get certain property from project folder."""
    with open(os.path.join(project, '__init__.py')) as f:
        result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                           f.read())
    return result.group(1)


setup(name=PROJECT,
      version=get_property('__version__', PROJECT),
      author='Ugur Cayoglu',
      package_data={PROJECT:['data/*']},
      author_email='cayoglu@kit.com',
      packages=find_packages(),
      )
