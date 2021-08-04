import numpy as np

import util
from environment import Environment, shape_cross
from water import Water


# points above and below to grow
POINTS_HEIGHT = 5
GROW_ENERGY = 5
INIT_ENERGY = 5
DELTA_ENERGY = 1
DROPS_TO_GROW = 5


class Plant:
    def __init__(self, size, percent=20):
        self.level = np.zeros(size, dtype=int)
        self.energy = np.zeros(size, dtype=int)
        # Fills level and energy
        self.add(percent)
        self.space = None

    def add(self, percent):
        # np.random.seed(23)
        size = self.level.size
        n_percent = size * percent // 100
        choices = np.random.choice(range(size), n_percent, replace=False)
        xs, ys = np.unravel_index(choices, self.level.shape)
        self.level[xs, ys] += 1
        self.energy[xs, ys] += INIT_ENERGY

    def groups(self):
        xs, ys = self.level.nonzero()
        already = np.zeros_like(self.level)
        groups = []
        for i, j in zip(xs, ys):
            if not already[i, j]:
                grp = util.groups((i, j), self.level)
                ixs, jys = zip(*grp)
                already[ixs, jys] = 1
                groups.append(grp)
        return groups

    def grow_hv(self, point):
        idxs = shape_cross(point, self.level.shape)
        level = self.space.surface[point] + self.level[point]

        for neigh in idxs:
            level_neigh = self.space.surface[neigh] + self.level[neigh]
            if abs(level - level_neigh) <= POINTS_HEIGHT:
                self.level[neigh] += 1
                self.energy[neigh] += INIT_ENERGY
                return
        self.level[point] += 1
        self.energy[point] += INIT_ENERGY

    def grow(self, water):
        groups = self.groups()
        for points in groups:
            drops = water.collect(points, POINTS_HEIGHT)
            qty_water = water.quantity(drops)

            for point in points:
                # Consume water each point at a time
                if qty_water > 0:
                    self.energy[point] += DELTA_ENERGY
                    qty_water -= DROPS_TO_GROW
                    reduced = DROPS_TO_GROW
                    if qty_water < 0:
                        reduced = qty_water + DROPS_TO_GROW
                    # reduce water around
                    water.reduce(drops, reduced)
                    min_grow = GROW_ENERGY * (self.level[point] + 1)
                    if self.energy[point] > min_grow:
                        self.grow_hv(point)
                else:   # water == 0 the plant loses energy or even die
                    self.energy[point] -= DELTA_ENERGY
                    min_decrease = GROW_ENERGY * (self.level[point] - 1)
                    if self.energy[point] <= min_decrease:
                        self.level[point] -= 1

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name} <{self.level!r}>'
