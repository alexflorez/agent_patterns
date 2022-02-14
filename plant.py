import numpy as np

import util


class Plant:
    INIT_LEVEL = 1
    INIT_ENERGY = 5
    # points above and below to grow
    POINTS_HEIGHT = 5
    GROW_ENERGY = 5
    DELTA_ENERGY = 1
    DROPS_TO_GROW = 5

    def __init__(self, shape):
        self.level = np.zeros(shape, dtype=int)
        self.energy = np.zeros(shape, dtype=int)

    def add(self, percent):
        shape = self.level.shape
        xs, ys = util.fill_region(shape, percent)
        self.level[xs, ys] += self.INIT_LEVEL
        self.energy[xs, ys] += self.INIT_ENERGY

    def add_values(self, positions, value):
        # positions is a list of points
        # [(0, 1), (1, 2), (2, 3)]
        x, y = zip(*positions)
        self.level[x, y] += value * self.INIT_LEVEL
        self.energy[x, y] += value * self.INIT_ENERGY

    def groups(self):
        xs, ys = self.level.nonzero()
        already = np.zeros_like(self.level)
        group_plants = []
        for point in zip(xs, ys):
            if not already[point]:
                grp = util.groups(point, self.level)
                ixs, jys = zip(*grp)
                already[ixs, jys] = 1
                group_plants.append(grp)
        return group_plants

    def grow_hv(self, space, point):
        indexes = util.shape_cross(point, self.level.shape)
        level = space.level[point] + self.level[point]

        for neighbor in indexes:
            level_neigh = space.level[neighbor] + self.level[neighbor]
            if abs(level - level_neigh) <= self.POINTS_HEIGHT:
                self.level[neighbor] += 1
                self.energy[neighbor] += self.INIT_ENERGY
                return
        self.level[point] += 1
        self.energy[point] += self.INIT_ENERGY

    def check_grow(self, space):
        groups_plant = self.groups()
        for group in groups_plant:
            pond = space.water.find(space, group, self.POINTS_HEIGHT)
            self.grow(space, group, pond)

    def grow(self, space, group, drops):
        qty_water = space.water.quantity(drops)
        for point in group:
            # Consume water each point at a time
            if qty_water > 0:
                self.energy[point] += self.DELTA_ENERGY
                qty_water -= self.DROPS_TO_GROW
                reduced = self.DROPS_TO_GROW
                if qty_water < 0:
                    reduced = qty_water + self.DROPS_TO_GROW
                # reduce water around
                space.water.reduce(drops, reduced)
                min_grow = self.GROW_ENERGY * (self.level[point] + 1)
                if self.energy[point] > min_grow:
                    self.grow_hv(space, point)
            else:   # water == 0 the plant loses energy or even die
                self.energy[point] -= self.DELTA_ENERGY
                min_decrease = self.GROW_ENERGY * (self.level[point] - 1)
                if self.energy[point] <= min_decrease:
                    self.level[point] -= 1

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name} <{self.level!r}>'


if __name__ == "__main__":
    from environment import Environment
    from sample_surfaces import data_slope
    from water import Water

    image = data_slope(6, 6)
    water = Water(image.shape)
    water.add(60)
    plant = Plant(image.shape)
    plant.add(30)
    env = Environment(image)
    env.water = water
    print(f"plant\n{plant.level}")
    print(f"env+plant\n{env.level + plant.level}")
    # print(f"water\n{water.level}")
    print(f"groups\n{plant.groups()}")
    print("\nPonds")
    plant.check_grow(env)
