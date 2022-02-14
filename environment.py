import numpy as np
import random
import functools


class Environment:
    def __init__(self, space):
        self.level = space.astype(int)
        self.water = None

    def surface(self):
        if self.water:
            return self.level + self.water.level
        return self.level

    def next_position(self, point):
        """ Computes the next position """
        random.seed(23)
        xs, ys = self.neighbors(point)
        data_space = self.surface()
        region = data_space[xs, ys]
        indexes = np.flatnonzero(region == region.min())
        index = random.choice(indexes)
        new_point = xs[index], ys[index]
        value = data_space[point]
        new_value = data_space[new_point]
        return new_point if value >= new_value else point

    @functools.lru_cache(maxsize=None)
    def neighbors(self, point):
        """
        8 1 2
        7 @ 3
        6 5 4
        """
        x, y = point
        xmax, ymax = self.level.shape
        if x == 0 and y == 0:
            # ((0, 1), (1, 1), (1, 0))
            xs = (x + 0, x + 1, x + 1)
            ys = (y + 1, y + 1, y + 0)
        elif x == 0 and y == ymax - 1:
            # ((1, 0), (1, -1), (0, -1))
            xs = (x + 1, x + 1, x + 0)
            ys = (y + 0, y - 1, y - 1)
        elif x == xmax - 1 and y == 0:
            # ((-1, 0), (-1, 1), (0, 1))
            xs = (x - 1, x - 1, x + 0)
            ys = (y + 0, y + 1, y + 1)
        elif x == xmax - 1 and y == ymax - 1:
            # ((-1, 0), (0, -1), (-1, -1))
            xs = (x - 1, x + 0, x - 1)
            ys = (y + 0, y - 1, y - 1)
        elif x == 0:
            # ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
            xs = (x + 0, x + 1, x + 1, x + 1, x + 0)
            ys = (y + 1, y + 1, y + 0, y - 1, y - 1)
        elif x == xmax - 1:
            # ((-1, 0), (-1, 1), (0, 1), (0, -1), (-1, -1))
            xs = (x - 1, x - 1, x + 0, x + 0, x - 1)
            ys = (y + 0, y + 1, y + 1, y - 1, y - 1)
        elif y == 0:
            # ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0))
            xs = (x - 1, x - 1, x + 0, x + 1, x + 1)
            ys = (y + 0, y + 1, y + 1, y + 1, y + 0)
        elif y == ymax - 1:
            # ((-1, 0), (1, 0), (1, -1), (0, -1), (-1, -1))
            xs = (x - 1, x + 1, x + 1, x + 0, x - 1)
            ys = (y + 0, y + 0, y - 1, y - 1, y - 1)
        else:
            # xy = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
            xs = (x - 1, x - 1, x + 0, x + 1, x + 1, x + 1, x + 0, x - 1)
            ys = (y + 0, y + 1, y + 1, y + 1, y + 0, y - 1, y - 1, y - 1)
        return xs, ys


if __name__ == "__main__":
    dt = np.arange(16).reshape(4, 4)
    print(dt)
    env = Environment(dt)
    from itertools import product
    for i, j in product(range(4), range(4)):
        pt = i, j
        ix, jy = env.neighbors(pt)
        print(f"{i, j}: {env.level[ix, jy]}")
