import numpy as np

class Water:
    def __init__(self, surface):
        self.surface = surface
        self.height = np.zeros_like(surface.level) + 1

    def add(self):
        self.height += 1

    def move(self, x, y):
        max_val = self.surface.level.max() + 1
        rows, columns = self.surface.shape
        ixs = []
        jys = []
        for i, j in product(range(-1, 2), range(-1, 2)):
            ix = (x + i + rows) % rows
            jy = (y + j + columns) % columns
            ixs.append(ix)
            jys.append(jy)
        

        x, y = w_surface.shape
        for i, j in product(range(1, x - 1), range(1, y - 1)):
            if self.height[i, j] >= 1:
                ni, nj = get_min(w_surface, self.height, i, j)
                self.height[i, j] -= 1
                self.height[ni, nj] += 1
        self.height = self.height[1:-1, 1:-1]
        return self.height

    def position(self):
        xs, ys = np.nonzero(self.height)
        zs = self.height[xs, ys]
        height = self.surface[xs, ys]
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

