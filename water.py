import numpy as np

import util


class Water:
    INIT_LEVEL = 1
    INIT_ENERGY = 10
    MOVE_LEVEL = 1
    MOVE_ENERGY = 1

    def __init__(self, shape, evaporate=False):
        self.level = np.zeros(shape, dtype=int)
        self.energy = np.zeros(shape, dtype=int)
        self.evaporate = evaporate

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

    def move(self, space):
        """ Move drops of water over level """
        xs, ys = self.level.nonzero()
        for position in zip(xs, ys):
            if self.evaporate:
                self.step_evaporate(space, position)
            else:
                self.step(space, position)

    def step(self, space, position):
        level = self.level[position]
        for _ in range(level):
            new_position = space.next_position(position)
            self.level[position] -= self.MOVE_LEVEL
            self.level[new_position] += self.MOVE_LEVEL

    def step_evaporate(self, space, position):
        divided = self.distribute_energy(position)
        for part_energy in divided:
            self.level[position] -= self.MOVE_LEVEL
            self.energy[position] -= part_energy

            tmp_energy = (part_energy - self.MOVE_ENERGY)
            if tmp_energy > 0:
                new_position = space.next_position(position)
                self.level[new_position] += self.MOVE_LEVEL
                self.energy[new_position] += tmp_energy

    def distribute_energy(self, position):
        energy = self.energy[position]
        level = self.level[position]
        divided = []
        k = energy // level
        r = energy % level
        for i in range(level):
            part = k
            if i == 0:
                part += r
            divided.append(part)
        return divided

    def allowed_height(self, space, point, height):
        indexes = util.shape_cross(point, self.level.shape)
        # including point itself
        indexes += [point]
        # filter neighbors with water
        indexes = [xy for xy in indexes if self.level[xy]]
        drops = []
        # height of plant at this point
        h_point = space.level[point]
        for neighbor in indexes:
            h_neighbor = space.level[neighbor]
            if h_point > h_neighbor:
                h_point += self.level[point]
                h_neighbor += self.level[neighbor]
            if abs(h_point - h_neighbor) <= height:
                drops.append(neighbor)
        return drops

    def find(self, space, points, height):
        """Find drops that contain water around given points"""
        already = np.zeros_like(self.level)
        water_region = set()
        for point in points:
            drops = self.allowed_height(space, point, height)
            for d in drops:
                if not already[d]:
                    grp = util.groups(d, self.level)
                    ixs, jys = zip(*grp)
                    already[ixs, jys] = 1
                    water_region |= grp
        return water_region

    def quantity(self, drops):
        """Returns the quantity of water around a region of drops"""
        if not drops:
            return 0
        xw, yw = zip(*drops)
        qty = self.level[xw, yw].sum()
        return qty

    def reduce(self, points, n_drops):
        """reduce n_drops of water around points"""
        while n_drops:
            for point in points:
                if self.level[point]:
                    self.level[point] -= 1
                    n_drops -= 1
                    if n_drops == 0:
                        break

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.level!r}>'


if __name__ == "__main__":
    from environment import Environment
    from util import plot_data
    from sample_surfaces import data_inv_pyramid

    image = data_inv_pyramid(100, 100)
    water = Water(image.shape, evaporate=True)
    water.add(100)
    # water.add_values([(2, 0)], 4)
    env = Environment(image)
    env.water = water

    iterations = 20
    water_dt = []
    energy_dt = []
    for i in range(iterations):
        water_dt.append(water.level.copy())
        energy_dt.append(water.energy.copy())
        # if i == 50:
        #     water.add(100)
        water.move(env)
        if (water.energy < 0).any():
            print(f"{water.energy.min()}")
    print(water.INIT_ENERGY)
    dt = np.array(water_dt)
    plot_data(dt, "water")
