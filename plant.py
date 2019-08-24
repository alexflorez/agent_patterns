import numpy as np
from itertools import product


class Plant:
    def __init__(self, surface, water):
        self.surface = surface
        self.water = water
        self.seeds = np.zeros_like(surface.level)

    def seed(self, percent):
        percent = self.seeds.size * percent // 100
        choices = np.random.choice(range(self.seeds.size), percent, replace=False)
        xs, ys = np.unravel_index(choices, self.seeds.shape)
        self.seeds[xs, ys] = self.seeds[xs, ys] + 1

    def region_idxs(self, x, y):
        rows, columns = self.surface.level.shape
        ixs = []
        jys = []
        for i, j in product(range(-1, 2), range(-1, 2)):
            ix = (x + i + rows) % rows
            jy = (y + j + columns) % columns
            ixs.append(ix)
            jys.append(jy)
        return ixs, jys
    
    def adjust_water(self, qty, xs, ys):
        while qty > 0:
            for x, y in zip(xs, ys):
                if self.water.height[x, y] > 0:
                    self.water.height[x, y] -= 1
                    qty -= 1
                if qty == 0:
                    break

    def grow(self, qty_grow):
        rows, columns = self.surface.level.shape
        for x, y in product(range(rows), range(columns)):
            if self.seeds[x, y] >= 1:
                ixs, jys = self.region_idxs(x, y)
                region = self.water.height[ixs, jys]
                qty_water = region.sum()
                if qty_water >= qty_grow:
                    # Vertical and horizontal growing
                    self.adjust_seeds(x, y)
                    # update water height
                    self.adjust_water(qty_grow, ixs, jys)
    
    def adjust_seeds(self, x, y):
        level_seed = self.surface.level[x, y] + self.seeds[x, y]
        ixs, jys = self.region_idxs(x, y)
        flag_horizontal = False
        for i, j in zip(ixs, jys):
            level_neighbor = self.surface.level[i, j] + self.seeds[i, j]
            # one level above and one level below
            if abs(level_seed - level_neighbor) == 1:
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
