import random
import numpy as np


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

    def minimal_idxs(self, tmp_height, x, y):
        """
        Find the minimum value indexes of a n x n region.
        If there is more than a minimum value, one is chosen at random.
        """
        # region surface and region height water
        ixs, jys = self.surface.region_idxs(x, y)
        reg_sf = self.surface.level[ixs, jys]
        reg_hw = tmp_height[ixs, jys]
        region = reg_sf + reg_hw
        # to randomly choose among the minimal values
        items, jtems = (region == min(region.ravel())).nonzero()
        nitems = len(items)
        k = random.randint(0, nitems - 1) if nitems > 1 else 0
        i, j = items[k], jtems[k]
        return ixs[i, j], jys[i, j]
        
    def move(self):
        """
        Update the position and height of the water.
        """
        nw_height = np.copy(self.height)
        xnz, ynz = self.height.nonzero()
        for x, y in zip(xnz, ynz):
            i, j = self.minimal_idxs(nw_height, x, y)
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
