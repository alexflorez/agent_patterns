from numba import jit
import random
import numpy as np


@jit(nopython=True, fastmath=True)
def min_idxs(region, ixs, jys, n):
    # Randomly choose a value less than the region's center
    c = n // 2
    items, jtems = np.nonzero(region < region[c, c])
    nitems = items.size
    if nitems:
        k = random.randint(0, nitems - 1)
        i, j = items[k], jtems[k]
        return ixs[i, j], jys[i, j]
    else:
        return ixs[c, c], jys[c, c]


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

    def set_points(self, xs, ys):
        self.height[xs, ys] = 1

    def set_data(self, data):
        self.height = data

    def drop(self, percent=50):
        xnz, ynz = self.height.nonzero()
        nz = len(xnz)
        qty_drop = nz * percent // 100
        ix = random.sample(range(nz), qty_drop)
        self.height[xnz[ix], ynz[ix]] -= 1

    def minimal_idxs(self, tmp_height, x, y):
        """
        Find the minimum value indexes of a n x n region.
        If there is more than a minimum value, one is chosen at random.
        """
        # region surface and region height water
        ixs, jys = self.surface.region_idxs(x, y)
        reg_sf = self.surface.level[ixs, jys]
        reg_wt = tmp_height[ixs, jys]
        region = reg_sf + reg_wt
        i, j = min_idxs(region, ixs, jys, self.surface.n_region)
        return i, j

    def move(self):
        """
        Update the position and height of the water.
        """
        nw_height = np.array(self.height)
        xnz, ynz = self.height.nonzero()
        for x, y in zip(xnz, ynz):
            i, j = self.minimal_idxs(nw_height, x, y)
            if (i, j) == (x, y):
                continue
            nw_height[x, y] -= 1
            nw_height[i, j] += 1
        self.height = nw_height

    def adjust_water(self, drops, ixs, jys):
        """
        Adjust the height of the water, subtract the drops
        until the specified amount is reduced.
        """
        while drops > 0:
            for x, y in zip(ixs.ravel(), jys.ravel()):
                if self.height[x, y] > 0:
                    self.height[x, y] -= 1
                    drops -= 1
                    if drops == 0:
                        break

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.height!r}>'
