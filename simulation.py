import numpy as np
from skimage import io


class Surface:
    def __init__(self, filename):
        self.surface = io.imread(filename, as_gray=True)


class Water:
    def __init__(self, level):
        self.level = np.copy(level)
        self.quantity = np.zeros_like(self.level)


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


if __name__ == '__main__':
    filename = "images/tinybeans.jpg"
    surface = Surface(filename)
    water = Water(surface.surface)
    seed = Seed(surface.surface, 10, 3)


