import numpy as np

class Plant:
    def __init__(self, surface, percent):
        self.surface = surface
        size = self.surface.size
        self.percent = size * percent // 100

    def seed(self):
        seeds = np.random.choice(range(size), self.percent, replace=False)
        xs, ys = np.unravel_index(seeds, self.surface.level.shape)
        