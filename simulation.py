import numpy as np
from skimage import io
from mayavi import mlab
from itertools import product


def fill_border(area, value):
    x, y = area.shape
    column = np.full(x, value)
    area = np.column_stack((column, area, column))
    x, y = area.shape
    row = np.full(y, value)
    area = np.vstack((row, area, row))
    return area


def get_min(level, quantity, i, j):
    neighbor = np.copy(level[i-1: i+2, j-1: j+2])
    qty = quantity[i-1: i+2, j-1: j+2]
    neighbor += qty
    ni, nj = np.unravel_index(np.argmin(neighbor, axis=None), neighbor.shape)
    return ni + i - 1, nj + j - 1


class Surface:
    def __init__(self, filename):
        self.surface = io.imread(filename, as_gray=True)
        self.surface = self.surface * 255
        self.surface = self.surface.astype(np.uint8)
        self.surface[self.surface > 200] -= 200
        self.surface[self.surface > 100] -= 100
        self.surface[self.surface > 50] -= 50
        self.surface[self.surface > 25] -= 25
        self.surface[self.surface > 15] -= 15

    def draw(self):
        x, y = self.surface.shape
        xline = range(x)
        yline = range(y)
        X, Y = np.meshgrid(xline, yline, indexing='ij')
        s = mlab.barchart(X, Y, self.surface, color=(1, 0.6, 0))


class Water:
    def __init__(self, level):
        self.level = np.copy(level)
        self.quantity = np.zeros_like(self.level)

    def add(self):
        self.quantity += 1

    def move(self):
        max_val = self.level.max() + 1
        w_level = fill_border(self.level, max_val)
        self.quantity = fill_border(self.quantity, max_val)
        x, y = w_level.shape
        for i, j in product(range(1, x - 1), range(1, y - 1)):
            if self.quantity[i, j] >= 1:
                ni, nj = get_min(w_level, self.quantity, i, j)
                self.quantity[i, j] -= 1
                self.quantity[ni, nj] += 1
        self.quantity = self.quantity[1:-1, 1:-1]
        return self.quantity

    def position(self):
        xs, ys = np.nonzero(self.quantity)
        zs = self.quantity[xs, ys]
        height = self.level[xs, ys]
        nhs = np.array([])
        nxs = np.array([])
        nys = np.array([])
        for x, y, z, h in zip(xs, ys, zs, height):
            nhs = np.append(nhs, np.arange(h + 1, h + z + 1))
            nxs = np.append(nxs, np.full(z, x))
            nys = np.append(nys, np.full(z, y))
        # minus 0.5 to visualize
        nhs -= 0.5
        return nxs, nys, nhs

    def draw(self):
        nxs, nys, nhs = self.position()
        self.waterpoints = mlab.points3d(nxs, nys, nhs,
                                         color=(0.6, 0.8, 1), scale_factor=0.9)


class Seed:
    def __init__(self, level, percent, height2consider):
        self.level = level
        size = level.size
        self.percent = size * percent // 100
        self.xs, self.ys, self.zs = self.sow()
        self.height2consider = height2consider

    def sow(self):
        size = self.level.size
        seeds = np.random.choice(range(size), self.percent, replace=False)
        xs, ys = np.unravel_index(seeds, self.level.shape)
        xys = ((x, y) for x, y in sorted(zip(xs, ys)))
        xs, ys = zip(*xys)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.ones(xs.size, dtype='int64')
        return xs, ys, zs

    def position(self):
        height = self.level[self.xs, self.ys]
        nhs = np.array([])
        nxs = np.array([])
        nys = np.array([])
        for x, y, z, h in zip(self.xs, self.ys, self.zs, height):
            nhs = np.append(nhs, np.arange(h + 1, h + z + 1))
            nxs = np.append(nxs, np.full(z, x))
            nys = np.append(nys, np.full(z, y))
        # minus 0.5 to visualize
        nhs -= 0.5
        return nxs, nys, nhs

    def draw(self):
        xs, ys, zs = self.position()
        self.points = mlab.points3d(xs, ys, zs, color=(0, 1, 0),
                                    mode='cube', scale_factor=0.9)

if __name__ == '__main__':
    filename = "images/tinybeans.jpg"
    surface = Surface(filename)
    surface.draw()
    water = Water(surface.surface)
    water.add()
    water.move()
    water.draw()
    seed = Seed(surface.surface, 10, 3)
    seed.draw()
    mlab.show()


