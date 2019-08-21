import random
import numpy as np
from itertools import product
from collections import OrderedDict


class Water:
    def __init__(self, surface):
        self.surface = surface
        self.height = np.zeros_like(surface.level) + 1

    def add(self):
        self.height += 1

    def region(self, x, y):
        rows, columns = self.surface.level.shape
        ixs = []
        jys = []
        for i, j in product(range(-1, 2), range(-1, 2)):
            ix = (x + i + rows) % rows
            jy = (y + j + columns) % columns
            ixs.append(ix)
            jys.append(jy)
        # get unique and ordered values from indexes
        ixs_uq = list(OrderedDict.fromkeys(ixs))
        jys_uq = list(OrderedDict.fromkeys(jys))
        return ixs, jys, self.surface.level[np.ix_(ixs_uq, jys_uq)]

    def minimal(self, x, y):
        ixs, jys, region = self.region(x, y)
        i = np.argmin(region, axis=None)
        return ixs[i], jys[i]

    def move(self):
        rows, columns = self.surface.level.shape
        height_mv = np.copy(self.height)
        for x, y in product(range(rows), range(columns)):
            if self.height[x, y] >= 1:
                i, j = self.minimal(x, y)
                height_mv[x, y] -= 1
                height_mv[i, j] += 1
        self.height = height_mv

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.height!r}>'
