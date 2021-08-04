import numpy as np

import util
from environment import Environment, shape_cross

ENERGY = 10


class Water:
    def __init__(self, size):
        self.level = np.zeros(size, dtype=int)
        self.energy = np.zeros(size, dtype=int)
        self.add()
        self.space = None

    def add(self):
        np.random.seed(23)
        values = np.random.randint(2, size=self.level.shape)
        self.level += values
        self.energy += values * ENERGY

    def move(self):
        xs, ys = self.level.nonzero()
        for i, j in zip(xs, ys):
            # run k times according to the level
            for k in range(self.level[i, j]):
                point = (i, j)
                energy = self.energy[point] - (self.level[point] - 1) * ENERGY
                self.energy[point] -= energy
                self.level[point] -= 1
                new_pos = self.space.next_position(point)
                # evaporation while moving
                self.energy[new_pos] += energy - 1
                self.level[new_pos] += 1
                if self.energy[new_pos] == 0:
                    self.level[new_pos] -= 1

    def allowed_height(self, point, height):
        idxs = shape_cross(point, self.level.shape)
        # including point itself
        idxs += [point]
        # filter neighbors with water
        idxs = [xy for xy in idxs if self.level[xy]]
        drops = []
        # height of this point, i.e. of the plant
        h_point = self.space.surface[point]
        for neigh in idxs:
            h_neigh = self.space.surface[neigh]
            if h_point > h_neigh:
                h_point += self.level[point]
                h_neigh += self.level[neigh]
            # w_point = self.level[point]
            # w_neigh = self.level[neigh]
            if abs(h_point - h_neigh) <= height:
                drops.append(neigh)
        return drops

    def collect(self, points, height):
        """Find drops that contain water around given points"""
        already = np.zeros_like(self.level)
        water_region = set()
        for point in points:
            drops = self.allowed_height(point, height)
            for d in drops:
                if not already[d]:
                    grp = util.groups(d, self.level)
                    ixs, jys = zip(*grp)
                    already[ixs, jys] = 1
                    water_region |= grp
        return water_region

    def quantity(self, drops):
        """Returns the quantity of water around a region of drops"""
        if not drops:
            return 0
        xw, yw = zip(*drops)
        qty = self.level[xw, yw].sum()
        return qty

    def reduce(self, points, n_drops):
        """reduce n_drops of water around points"""
        while n_drops:
            for point in points:
                if self.level[point]:
                    self.level[point] -= 1
                    n_drops -= 1
                    if n_drops == 0:
                        break

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.level!r}>'
