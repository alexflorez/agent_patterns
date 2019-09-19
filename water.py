import random
import numpy as np
from itertools import product
from collections import OrderedDict
import functools
import array


class Water:
    def __init__(self, surface):
        self.surface = surface
        self.height = np.zeros_like(surface.level)

    def add(self, percent=100):
        """
        Add water to the surface level.
        Water can be added in percentages.
        """
        if percent == 100:
            self.height = self.height + 1
            return
        percent = self.height.size * percent // 100
        choices = random.sample(range(self.height.size), percent)
        xs, ys = np.unravel_index(choices, self.height.shape)
        self.height[xs, ys] = self.height[xs, ys] + 1

    @functools.lru_cache(maxsize=None)
    def region_idxs_unq(self, x, y):
        """
        Find the unique values of the indexes of a 3x3 region.
        """
        rows, columns = self.height.shape
        ixs, jys = self.surface.region_idxs(x, y, rows, columns)
        # get unique and ordered values from indexes
        ixs_uq = array.array('B', OrderedDict.fromkeys(ixs))
        jys_uq = array.array('B', OrderedDict.fromkeys(jys))
        return ixs, jys, ixs_uq, jys_uq

    def minimal(self, tmp_height, x, y):
        """
        Find the minimum value indexes of a 3x3 region.
        If there is more than a minimum value, one is chosen at random.
        """
        ixs, jys, ixs_uq, jys_uq = self.region_idxs_unq(x, y)
        # region surface and region height water
        # changing np.ix_(ixs_uq, jys_uq)
        ix = np.array(ixs_uq)[np.newaxis].T
        jy = np.array(jys_uq)[np.newaxis]
        reg_sf = self.surface.level[ix, jy]
        reg_hw = tmp_height[ix, jy]
        region = reg_sf + reg_hw
        # to randomly choose among the minimal values
        # i = np.nonzero(region.ravel() == region.min())[0]
        items = (region.ravel() == region.min()).nonzero()[0]
        if len(items) > 1:
            k = random.choice(items)
        else:
            k = items[0]
        return ixs[k], jys[k]
        
    def move(self):
        """
        Update the position and height of the water.
        """
        rows, columns = self.height.shape
        nw_height = np.copy(self.height)
        for x, y in product(range(rows), range(columns)):
            if self.height[x, y] >= 1:
                i, j = self.minimal(nw_height, x, y)
                nw_height[x, y] -= 1
                nw_height[i, j] += 1
        self.height = nw_height

    def adjust_water(self, drops, xs, ys):
        """
        Adjust the height of the water, subtract the drops
        until the specified amount is reduced.
        """
        while drops > 0:
            for x, y in zip(xs, ys):
                if self.height[x, y] > 0:
                    self.height[x, y] -= 1
                    drops -= 1
                if drops == 0:
                    break

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.height!r}>'
