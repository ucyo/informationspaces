#!/usr/bin/env python
#!/usr/bin/env python
# coding: utf-8
"""
Quality assessment of the residue array given as input.
"""

from pasc.toolbox import get_bits
import numpy as np


class QA(object):
    """
    Quality Assessment of the Residue array
    """

    def __init__(self, residuearray):
        self.rarray = residuearray
        self.bits = get_bits(residuearray.array)

    @property
    def size(self):
        return self.rarray.array.size

    @property
    def nbytes(self):
        return self.size * self.bits

    def lzc(self, percent=True):
        lzc = lzcu(self.rarray.array, self.bits).sum()
        if percent:
            result = (lzc / self.nbytes) * 100
        else:
            result = lzc
        return result

    def tzc(self, percent=True):
        tzc = tzcu(self.rarray.array, self.bits).sum()
        if percent:
            result = (tzc / self.nbytes) * 100
        else:
            result = tzc
        return result

    @property
    def lzcND(self):
        return lzcu(self.rarray.array, self.bits).astype(int)

    @property
    def tzcND(self):
        return tzcu(self.rarray.array, self.bits).astype(int)

    def __repr__(self):
        res = "QA: {} ({:.2f}%) LZC, {} ({:.2f}%) TZC".format(
            self.lzc(False), self.lzc(),
            self.tzc(False), self.tzc())
        return res


def tzc(val, bits=32):
    """Count trailing zeroes."""
    cnt = 0
    for i in range(0, bits):
        if val & (1 << i) != 0:
            break
        cnt += 1
    return cnt


tzcu = np.frompyfunc(tzc, 2, 1)


def lzc(val, bits=32):
    """Count leading zeroes."""
    cnt = 0
    for i in range(0, bits):
        if val & (1 << (bits - 1 - i)) != 0:
            break
        cnt += 1
    return cnt


lzcu = np.frompyfunc(lzc, 2, 1)
