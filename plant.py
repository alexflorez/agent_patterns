import numpy as np
from itertools import product
from collections import OrderedDict


class Plant:
    def __init__(self, surface, water, percent):
        self.surface = surface
        self.water = water
        self.seeds = np.zeros_like(surface.level)
        self.percent = self.seeds.size * percent // 100

    def seed(self):
        choices = np.random.choice(range(self.seeds.size), self.percent, replace=False)
        xs, ys = np.unravel_index(choices, self.seeds.level.shape)
        self.seeds[xs, ys] = self.seeds[xs, ys] + 1

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
        return ixs, jys, self.water.height[np.ix_(ixs_uq, jys_uq)]

    def check(self, x, y):
        ixs, jys, region = self.region(x, y)
        drops = region.sum()
        return drops
    
    def grow(self):
        N = 5
        rows, columns = self.surface.level.shape
        seeds_mv = np.copy(self.seeds)
        for x, y in product(range(rows), range(columns)):
            if self.seeds[x, y] >= 1:
                drops = self.check(x, y)
                if drops > N:
                    seeds_mv[i, j] += 1
                else:
                    seeds_mv[x, y] -= 1
        self.seeds = seeds_mv
    
    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.seeds!r}>'
