import random
import numpy as np


class Plant:
    # points pixel above and points pixel below to grow
    POINTS = 10      # difference (level + seed) - neighbor(level + seed)
    INIT_ENERGY = 4
    DELTA_INCREASE = 1
    DELTA_DECREASE = 1
    DECREASE = 0
    GROWTH = 4

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
        # random.seed(21)
        percent = self.seeds.size * percent // 100
        choices = random.sample(range(self.seeds.size), percent)
        xs, ys = np.unravel_index(choices, self.seeds.shape)
        self.seeds[xs, ys] += 1
        self.energy[xs, ys] += self.INIT_ENERGY

    def set_points(self, xs, ys):
        self.seeds[xs, ys] = 1

    def set_data(self, data):
        self.seeds = data

    def qty_water_region(self, x, y):
        ixs, jys = self.surface.region_idxs(x, y)
        region = self.water.height[ixs, jys]
        return ixs, jys, sum(region.ravel())

    def horiz_vert_grow(self, qty_water, x, y):
        # consume water in this position
        # each drop provides 1 unit of energy
        energy = qty_water * self.DELTA_INCREASE
        self.energy[x, y] += energy
        qty_growth = self.GROWTH * self.seeds[x, y]
        if self.energy[x, y] >= qty_growth:
            # Vertical and horizontal growth
            self.adjust_seeds(x, y)

    def grow_by_points(self):
        """
        Assess the plant growth according to
        the amount of water around a n x n region.
        """
        xseeds, yseeds = np.nonzero(self.seeds)
        for x, y in zip(xseeds, yseeds):
            ixs, jys, qty_water = self.qty_water_region(x, y)
            if qty_water:
                self.horiz_vert_grow(qty_water, x, y)
                # Update water height
                self.water.adjust_water(qty_water, ixs, jys)
            else:   
                # check if neighbors have access to water
                neigh = self.neighbors(x, y)
                for i, j in neigh:
                    nxs, nys, qty_water = self.qty_water_region(i, j)
                    if qty_water:
                        self.horiz_vert_grow(qty_water, x, y)
                        # Update water height
                        self.water.adjust_water(qty_water, nxs, nys)
                        break
                else:                
                    self.energy[x, y] -= self.DELTA_DECREASE
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
                self.energy[x, y] -= self.DELTA_DECREASE
                break
    
        if not flag_horizontal:
            # Vertical growing.
            self.seeds[x, y] += 1
            # add INIT_ENERGY, subtract DELTA_ENERGY
            self.energy[x, y] += self.INIT_ENERGY
            self.energy[x, y] -= self.DELTA_DECREASE
    
    def neighbors(self, x, y):
        start = (x, y)
        visited, stack = set(), [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                i, j = vertex
                ixs, jys = self.surface.region_idxs(i, j)
                xpts, ypts = self.seeds[ixs, jys].nonzero()
                nodes = {(ixs[i, j], jys[i, j]) for i, j in zip(xpts, ypts)}
                stack.extend(nodes - visited)
        # comment for grow by groups
        # visited.remove(start)
        return visited

    def find_groups(self):
        xs, ys = self.seeds.nonzero()
        visited = np.zeros_like(self.seeds, dtype=np.uint8)
        groups = []
        for x, y in zip(xs, ys):
            if visited[x, y] == 0:
                neighs = self.neighbors(x, y)
                neighs |= {(x, y)}
                ixs, jys = zip(*neighs)
                visited[ixs, jys] = 1
                groups.append(neighs)
        return groups

    def water_region(self, group):
        water = set()
        for x, y in group:
            ixs, jys = self.surface.region_idxs(x, y)
            water |= set(zip(ixs.ravel(), jys.ravel()))
        return water

    def hv_grow(self, qty, x, y):
        """
        Returns the quantity of water used
        """
        if qty >= self.GROWTH:
            self.horiz_vert_grow(self.GROWTH, x, y)
            return self.GROWTH
        else:
            self.horiz_vert_grow(qty, x, y)
            return qty

    def grow_by_groups(self):
        groups_plant = self.find_groups()
        for group in groups_plant:
            water = self.water_region(group)
            real_water = [(x, y) for x, y in water if self.water.height[x, y]]
            if real_water:
                xs, ys = zip(*real_water)
                region_water = self.water.height[xs, ys]
                qty_water = region_water.sum()
            else:
                qty_water = 0

            for x, y in group:
                if qty_water:
                    qty_decreased = self.hv_grow(qty_water, x, y)
                    self.water.reduce_water(qty_decreased, xs, ys)
                    qty_water -= qty_decreased
                else:
                    self.energy[x, y] -= self.DELTA_DECREASE
                    qty_decrease = self.GROWTH * (self.seeds[x, y] - 1)
                    if self.energy[x, y] <= qty_decrease:
                        self.seeds[x, y] -= 1

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.seeds!r}>'
