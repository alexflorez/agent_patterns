import numpy as np
from collections import namedtuple
Point = namedtuple('Point', 'x y')


def shape_neighborhood(index):
    """
    # # #
    # @ #
    # # #
    """
    # x == 0 and y == 0:
    x0y0 = ((0, 1), (1, 1), (1, 0))
    # x == 0 and y == ymax - 1:
    x0yM = ((1, 0), (1, -1), (0, -1))
    # x == xmax - 1 and y == 0:
    xMy0 = ((-1, 0), (-1, 1), (0, 1))
    # x == xmax - 1 and y == ymax - 1:
    xMyM = ((-1, 0), (0, -1), (-1, -1))
    # x == 0:
    x0y = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    # x == xmax - 1:
    xMy = ((-1, 0), (-1, 1), (0, 1), (0, -1), (-1, -1))
    # y == 0:
    xy0 = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0))
    # y == ymax - 1:
    xyM = ((-1, 0), (1, 0), (1, -1), (0, -1), (-1, -1))
    # x, y
    xy = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    shape = [x0y0, x0yM, xMy0, xMyM, x0y, xMy, xy0, xyM, xy]
    return shape[index]


def shape_cross(point, shape):
    """
      #
    # @ #
      #
    """
    x, y = point
    xmax, ymax = shape
    if x == 0 and y == 0:
        return [(x, y + 1), (x + 1, y)]
        # return (x, x + 1), (y + 1, y)
    elif x == 0 and y == ymax - 1:
        return [(x + 1, y), (x, y - 1)]
        # return (x + 1, x), (y, y - 1)
    elif x == xmax - 1 and y == 0:
        return [(x - 1, y), (x, y + 1)]
        # return (x - 1, x), (y, y + 1)
    elif x == xmax - 1 and y == ymax - 1:
        return [(x - 1, y), (x, y - 1)]
        # return (x - 1, x), (y, y - 1)
    elif x == 0:
        return [(x, y + 1), (x + 1, y), (x, y - 1)]
        # return (x, x + 1, x), (y + 1, y, y - 1)
    elif x == xmax - 1:
        return [(x - 1, y), (x, y + 1), (x, y - 1)]
        # return (x - 1, x, x), (y, y + 1, y - 1)
    elif y == 0:
        return [(x - 1, y), (x, y + 1), (x + 1, y)]
        # return (x - 1, x, x + 1), (y, y + 1, y)
    elif y == ymax - 1:
        return [(x - 1, y), (x + 1, y), (x, y - 1)]
        # return (x - 1, x + 1, x), (y, y, y - 1)
    else:  # x, y
        return [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        # return (x - 1, x, x + 1, x), (y, y + 1, y, y - 1)


class Environment:
    def __init__(self, surface, water):
        self.surface = surface
        self.water = water

    def possibles(self, point):
        region, shape = self.neighbors(point)
        index = np.argmin(region)
        return index, shape

    def value(self, point):
        return self.surface[point] + self.water.level[point]

    def update(self, a_point, n_point):
        self.water.level[a_point] -= 1
        self.water.level[n_point] += 1

    def neighbors(self, point):
        x, y = point
        data = self.surface + self.water.level
        xmax, ymax = data.shape
        if x == 0 and y == 0:
            idx = 0
            region = np.r_[data[x:x+2, y+1], data[x+1, y]]
        elif x == 0 and y == ymax - 1:
            idx = 1
            region = np.r_[data[x+1, y], data[x:x+2, y-1][::-1]]
        elif x == xmax - 1 and y == 0:
            idx = 2
            region = np.r_[data[x-1, y], data[x-1:x+1, y+1]]
        elif x == xmax - 1 and y == ymax - 1:
            idx = 3
            region = np.r_[data[x-1, y], data[x-1:x+1, y-1][::-1]]
        elif x == 0:
            idx = 4
            region = np.r_[data[x:x+2, y+1], data[x+1, y], data[x:x+2, y-1][::-1]]
        elif x == xmax - 1:
            idx = 5
            region = np.r_[data[x-1, y], data[x-1:x+1, y+1], data[x-1:x+1, y-1][::-1]]
        elif y == 0:
            idx = 6
            region = np.r_[data[x-1, y], data[x-1:x+2, y+1], data[x+1, y]]
        elif y == ymax - 1:
            idx = 7
            region = np.r_[data[x-1, y], data[x+1, y], data[x-1:x+2, y-1][::-1]]
        else:
            idx = 8
            region = np.r_[data[x-1, y], data[x-1:x+2, y+1], data[x+1, y], data[x-1:x+2, y-1][::-1]]
        return np.array(region, dtype=int), shape_neighborhood(idx)
