import random
import numpy as np


class Plant:
    # points pixel above and points pixel below to grow
    POINTS = 3      # difference (level + seed) - neighbor(level + seed)
    INIT_ENERGY = 0.5
    DELTA_ENERGY = 0.1
    DECREASE = 0.0
    GROWTH = 1.0

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
        #random.seed(5)
        percent = self.seeds.size * percent // 100
        choices = random.sample(range(self.seeds.size), percent)
        xs, ys = np.unravel_index(choices, self.seeds.shape)
        self.seeds[xs, ys] += 1
        self.energy[xs, ys] += self.INIT_ENERGY

    def qty_water_region(self, x, y):
        rows, columns = self.surface.level.shape
        ixs, jys = self.surface.region_idxs(x, y, rows, columns)
        region = self.water.height[ixs, jys]
        return ixs, jys, region.sum()
    
    def horiz_vert_grow(self, x, y):
        self.energy[x, y] += self.DELTA_ENERGY
        qty_growth = self.GROWTH * self.seeds[x, y]
        if self.energy[x, y] >= qty_growth:
            # Vertical and horizontal growth
            self.adjust_seeds(x, y)

    def grow(self, qty_water_grow):
        """
        Assess the plant growth according to
        the amount of water around a 3x3 region.
        """
        xseeds, yseeds = np.nonzero(self.seeds)
        for x, y in zip(xseeds, yseeds):
            ixs, jys, qty_water = self.qty_water_region(x, y)
            if qty_water >= qty_water_grow:
                self.horiz_vert_grow(x, y)
                # Update water height
                self.water.adjust_water(qty_water_grow, ixs, jys)
            else:   
                # check if neigbors have qty_water_grow
                neigh = set()
                neigh = self.neighbors(neigh, x, y)
                neigh = neigh - {(x, y)}
                
                for i, j in neigh:
                    nxs, nys, qty_water = self.qty_water_region(i, j)
                    if qty_water >= qty_water_grow:
                        self.horiz_vert_grow(x, y)
                        # Update water height
                        self.water.adjust_water(qty_water_grow, nxs, nys)
                        break
                else:                
                    self.energy[x, y] -= self.DELTA_ENERGY
                    qty_decrease = self.INIT_ENERGY * self.seeds[x, y]
                    if self.energy[x, y] < qty_decrease:
                        self.seeds[x, y] -= 1
    
    def adjust_seeds(self, x, y):
        """
        Update plant growth in vertical and horizontal ways.
        """
        rows, columns = self.surface.level.shape        
        level_seed = self.surface.level[x, y] + self.seeds[x, y]
        level_seed = float(level_seed)
        ixs, jys = self.surface.region_idxs(x, y, rows, columns)
        flag_horizontal = False
        # only consider neighbors not the same point
        cixs = ixs[:]
        cjys = jys[:]
        m = len(cixs) // 2
        del cixs[m]
        del cjys[m]
        for i, j in zip(cixs, cjys):
            level_neighbor = self.surface.level[i, j] + self.seeds[i, j]
            level_neighbor = float(level_neighbor)
            allowed_heights = list(range(1, self.POINTS + 1))
            if abs(level_seed - level_neighbor) in allowed_heights:
                # Horizontal growing
                self.seeds[i, j] += 1
                self.energy[i, j] += self.INIT_ENERGY
                flag_horizontal = True
                # Because of growing, lose some energy
                self.energy[x, y] -= self.DELTA_ENERGY
                break
        if not flag_horizontal:
            # Vertical growing.
            self.seeds[x, y] += 1
            # add INIT_ENERGY, subtract DELTA_ENERGY
            self.energy[x, y] += self.INIT_ENERGY
            self.energy[x, y] -= self.DELTA_ENERGY

    def neighbors(self, neigh, x, y):
        if (x, y) in neigh:
            return set()
        rows, columns = self.seeds.shape
        ixs, jys = self.surface.region_idxs(x, y, rows, columns)
                
        pts = np.nonzero(self.seeds[ixs, jys])[0]

        neigh = neigh | {(x, y)}
        for p in pts:
            neigh = neigh | self.neighbors(neigh, ixs[p], jys[p])
        return neigh

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.seeds!r}>'
