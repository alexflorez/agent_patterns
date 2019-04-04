import numpy as np
from skimage import io
from mayavi import mlab


class Surface:
    def __init__(self, filename):
        self.surface = io.imread(filename, as_gray=True)
        self.surface = self.surface * 255
        self.surface = self.surface.astype(np.uint8)

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
    water.draw()
    seed = Seed(surface.surface, 10, 3)
    seed.draw()
    mlab.show()


