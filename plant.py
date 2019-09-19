import random
import numpy as np


class Plant:
    # points pixel above and points pixel below to grow
    POINTS = 2
    INIT_ENERGY = 0.5
    DELTA_ENERGY = 0.1
    DECREASE = 0
    GROWTH = 1

    def __init__(self, surface, water):
        self.surface = surface
        self.water = water
        self.seeds = np.zeros_like(surface.level)
        self.energy = np.zeros_like(surface.level, dtype=np.float32)

    def seed(self, percent):
        """
        Add seeds to the surface level.
        Seeds are added over the surface in percentages.
        """
        percent = self.seeds.size * percent // 100
        choices = random.sample(range(self.seeds.size), percent)
        xs, ys = np.unravel_index(choices, self.seeds.shape)
        self.seeds[xs, ys] = self.seeds[xs, ys] + 1
        self.energy[xs, ys] = self.energy[xs, ys] + self.INIT_ENERGY

    def grow(self, qty_grow):
        """
        Assess the plant growth according to
        the amount of water around a 3x3 region.
        """
        rows, columns = self.surface.level.shape
        xseeds, yseeds = np.nonzero(self.seeds)
        for x, y in zip(xseeds, yseeds):
            ixs, jys = self.surface.region_idxs(x, y, rows, columns)
            region = self.water.height[ixs, jys]
            qty_water = region.sum()
            if qty_water >= qty_grow:
                self.energy[x, y] += self.DELTA_ENERGY
                if self.energy[x, y] >= self.GROWTH:
                    # Vertical and horizontal growth
                    self.adjust_seeds(x, y)
                # Update water height
                self.water.adjust_water(qty_grow, ixs, jys)
            else:   # qty_water less than qty_grow
                self.energy[x, y] -= self.DELTA_ENERGY
                if self.energy[x, y] <= self.DECREASE:
                    self.seeds[x, y] = self.seeds[x, y] - 1
    
    def adjust_seeds(self, x, y):
        """
        Update plant growth in vertical and horizontal ways.
        """
        rows, columns = self.surface.level.shape        
        level_seed = self.surface.level[x, y] + self.seeds[x, y]
        level_seed = float(level_seed)
        ixs, jys = self.surface.region_idxs(x, y, rows, columns)
        flag_horizontal = False
        for i, j in zip(ixs, jys):
            level_neighbor = self.surface.level[i, j] + self.seeds[i, j]
            level_neighbor = float(level_neighbor)
            if abs(level_seed - level_neighbor) == self.POINTS:
                # Horizontal growing
                self.seeds[i, j] += 1
                self.energy[i, j] = self.INIT_ENERGY
                flag_horizontal = True
                break
        if not flag_horizontal:
            # Vertical growing.
            self.seeds[x, y] += 1
            self.energy[x, y] = self.INIT_ENERGY
    
    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.seeds!r}>'
