import util
from environment import Environment, Point


class Raindrop:
    def __init__(self, space, position, energy):
        self.space = space
        self.position = position
        self.minimum = False
        self.energy = energy

    def move(self):
        while self.energy > 0 and not self.minimum:
            self.step()
            self.energy -= 1

    def step(self):
        index, shape = self.space.possibles(self.position)
        dx, dy = shape[index]
        x, y = self.position
        new_pos = Point(x + dx, y + dy)
        value = self.space.value(self.position)
        new_value = self.space.value(new_pos)
        if value - new_value > 1:
            self.position = new_pos
        else:
            self.minimum = True

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name} <{self.position!r}>'


if __name__ == "__main__":
    file = "images/c001_004.png"
    image = util.read(file)
    m, n = image.shape
    from water import Water
    water = Water(image.shape)
    env = Environment(image, water)
    water.space = env

    ini_energy = 4
    xs, ys = water.level.nonzero()
    for i, j in zip(xs, ys):
        point = Point(i, j)
        drop = Raindrop(env, point, ini_energy)
        print(f"Start: {env.surface[drop.position]}, Water: {water.level[drop.position]} at {drop.position}")
        while drop.energy and not drop.minimum:
            drop.step()
            print(f"\tHeight: {env.surface[drop.position]}, Water: {water.level[drop.position]} at {drop.position}")
