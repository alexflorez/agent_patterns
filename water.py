import numpy as np

import util
from raindrop import Raindrop
from environment import Environment, Point, shape_cross

ENERGY = 10


class Water:
    def __init__(self, size):
        self.level = np.zeros(size, dtype=int)
        self.energy = np.zeros(size, dtype=int)
        self.add()
        self.space = None

    def add(self):
        # np.random.seed(23)
        values = np.random.randint(2, size=self.level.shape)
        self.level += values
        self.energy += values * ENERGY

    def move(self):
        xs, ys = self.level.nonzero()
        for i, j in zip(xs, ys):
            # run k times according to the level
            for k in range(self.level[i, j]):
                point = Point(i, j)
                energy = self.energy[point] - (self.level[point] - 1) * ENERGY
                self.energy[point] -= energy
                self.level[point] -= 1
                raindrop = Raindrop(self.space, point, energy)
                raindrop.move()
                self.update(raindrop)

    def update(self, raindrop):
        if raindrop.energy > 0:
            new_pos = raindrop.position
            self.energy[new_pos] += raindrop.energy
            self.level[new_pos] += 1

    def allowed_height(self, point, height):
        idxs = shape_cross(point, self.level.shape)
        # including point itself
        idxs += [point]
        # filter neighbors with water
        idxs = [xy for xy in idxs if self.level[xy]]
        drops = []
        # height of this point, i.e. of the plant
        h_point = self.space.surface[point].astype(int)
        for neigh in idxs:
            h_neigh = self.space.surface[neigh].astype(int)
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


if __name__ == "__main__":
    file = "images/tinybeans.jpg"
    # file = "images/c001_004.png"
    # image = util.read(file, resize=False)

    image = np.array([[1, 2, 4, 3, 2, 1],
                      [5, 2, 2, 1, 7, 1],
                      [2, 2, 1, 7, 3, 2],
                      [8, 6, 1, 3, 2, 4],
                      [2, 1, 2, 5, 2, 3],
                      [7, 5, 2, 2, 1, 4]])
    water = Water(image.shape)
    space = Environment(image, water)
    water.space = space
    iterations = 10
    for _ in range(iterations):
        water.move()
        # water.add()
        xl, yl = np.nonzero(water.level)
        xe, ye = np.nonzero(water.energy)
        assert np.all(xl == xe) and np.all(yl == ye)
