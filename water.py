import numpy as np
from itertools import product
from collections import OrderedDict


class Water:
    def __init__(self, surface):
        self.surface = surface
        self.height = np.zeros_like(surface.level) + 1

    def add(self):
        self.height += 1

    def region_idxs(self, x, y):
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
        return ixs, jys, ixs_uq, jys_uq

    def minimal(self, tmp_height, x, y):
        ixs, jys, ixs_uq, jys_uq = self.region_idxs(x, y)
        # region surface and region hight water
        reg_sf = self.surface.level[np.ix_(ixs_uq, jys_uq)]
        reg_hw = tmp_height[np.ix_(ixs_uq, jys_uq)]
        region = reg_sf + reg_hw
        i = np.argmin(region, axis=None)
        return ixs[i], jys[i]

    def move(self):
        rows, columns = self.surface.level.shape
        nw_height = np.copy(self.height)
        for x, y in product(range(rows), range(columns)):
            if self.height[x, y] >= 1:
                i, j = self.minimal(nw_height, x, y)
                nw_height[x, y] -= 1
                nw_height[i, j] += 1
        self.height = nw_height

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.height!r}>'
