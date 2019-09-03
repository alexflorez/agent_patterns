import numpy as np
from itertools import product


class Plant:
    # points pixel above and points pixel below to grow
    POINTS = 1

    def __init__(self, surface, water):
        self.surface = surface
        self.water = water
        self.seeds = np.zeros_like(surface.level)

    def seed(self, percent):
        """
        Add seeds to the surface level.
        Seeds are added over the surface in percentages.
        """
        percent = self.seeds.size * percent // 100
        choices = np.random.choice(range(self.seeds.size), percent, replace=False)
        xs, ys = np.unravel_index(choices, self.seeds.shape)
        self.seeds[xs, ys] = self.seeds[xs, ys] + 1

    def grow(self, qty_grow):
        """
        Assess the plant growth according to
        the amount of water around a 3x3 region.
        """
        rows, columns = self.surface.level.shape
        for x, y in product(range(rows), range(columns)):
            if self.seeds[x, y] >= 1:
                ixs, jys = self.surface.region_idxs(x, y)
                region = self.water.height[ixs, jys]
                qty_water = region.sum()
                if qty_water >= qty_grow:
                    # Vertical and horizontal growth
                    self.adjust_seeds(x, y)
                    # Update water height
                    self.water.adjust_water(qty_grow, ixs, jys)
    
    def adjust_seeds(self, x, y):
        """
        Update plant growth in vertical and horizontal ways.
        """
        level_seed = self.surface.level[x, y] + self.seeds[x, y]
        ixs, jys = self.surface.region_idxs(x, y)
        flag_horizontal = False
        for i, j in zip(ixs, jys):
            level_neighbor = self.surface.level[i, j] + self.seeds[i, j]
            if abs(level_seed - level_neighbor) == self.POINTS:
                # Horizontal growing
                self.seeds[i, j] += 1
                flag_horizontal = True
                break
        if not flag_horizontal:
            # Vertical growing.
            self.seeds[x, y] += 1
    
    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.seeds!r}>'
