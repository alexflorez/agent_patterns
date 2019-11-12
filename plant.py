import random
import numpy as np


class Plant:
    # points pixel above and points pixel below to grow
    POINTS = 10      # difference (level + seed) - neighbor(level + seed)
    INIT_ENERGY = 5
    DELTA_ENERGY = 1
    DECREASE = 0
    GROWTH = 10

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
        random.seed(21)
        percent = self.seeds.size * percent // 100
        choices = random.sample(range(self.seeds.size), percent)
        xs, ys = np.unravel_index(choices, self.seeds.shape)
        self.seeds[xs, ys] += 1
        self.energy[xs, ys] += self.INIT_ENERGY

    def qty_water_region(self, x, y):
        ixs, jys = self.surface.region_idxs(x, y)
        region = self.water.height[ixs, jys]
        return ixs, jys, sum(region.ravel())
    
    def horiz_vert_grow(self, x, y):
        self.energy[x, y] += self.DELTA_ENERGY
        qty_growth = self.GROWTH * self.seeds[x, y]
        if self.energy[x, y] >= qty_growth:
            # Vertical and horizontal growth
            self.adjust_seeds(x, y)

    def grow(self, qty_water_grow):
        """
        Assess the plant growth according to
        the amount of water around a n x n region.
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
                neigh = self.neighbors(x, y)
                for i, j in neigh:
                    nxs, nys, qty_water = self.qty_water_region(i, j)
                    if qty_water >= qty_water_grow:
                        self.horiz_vert_grow(x, y)
                        # Update water height
                        self.water.adjust_water(qty_water_grow, nxs, nys)
                        break
                else:                
                    self.energy[x, y] -= self.DELTA_ENERGY
                    qty_decrease = self.GROWTH * (self.seeds[x, y] - 1)
                    if self.energy[x, y] <= qty_decrease:
                        self.seeds[x, y] -= 1
    
    def adjust_seeds(self, x, y):
        """
        Update plant growth in vertical and horizontal ways.
        """
        level_seed = self.surface.level[x, y] + self.seeds[x, y]
        level_seed = float(level_seed)
        ixs, jys = self.surface.region_idxs(x, y)
        flag_horizontal = False
        for i, j in zip(ixs.ravel(), jys.ravel()):
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

    def neighbors(self, x, y):
        node = (x, y)
        visited = {node}
        stack = [node]
        while stack:
            node = stack[-1]
            if node not in visited:
                visited.add(node)
            remove_from_stack = True
            i, j = node
            ixs, jys = self.surface.region_idxs(i, j)
            xpts, ypts = self.seeds[ixs, jys].nonzero()
            for i, j in zip(xpts, ypts):
                new_node = (ixs[i, j], jys[i, j])
                if new_node not in visited:
                    stack.append(new_node)
                    remove_from_stack = False
                    break
            if remove_from_stack:
                stack.pop()
        visited.remove((x, y))
        return visited

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.seeds!r}>'
