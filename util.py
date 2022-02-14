import numpy as np

from skimage import io
from skimage import transform
from skimage.util import img_as_ubyte
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def groups(point, space):
    """Starting from a point find a group of neighbors"""
    neighbors = {point}
    visited = deque()
    visited.append(point)
    while visited:
        xy = visited.pop()
        indexes = shape_cross(xy, space.shape)
        ns = {ij for ij in indexes if space[ij]}
        visited.extend(ns - neighbors)
        neighbors |= ns
    return neighbors


def fill_region(shape, percent):
    np.random.seed(23)
    m, n = shape
    size = m * n
    n_points = size * percent // 100
    choices = np.random.choice(range(size), n_points, replace=False)
    xs, ys = np.unravel_index(choices, shape)
    return xs, ys


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


def read(filename):
    """ png files are in the range [0 255] np.uint8
        jpg files are in the range [0 1] np.float
    """
    image = io.imread(filename, as_gray=True)
    if image.dtype == np.float:
        image = img_as_ubyte(image)
    return image


def resize(image, new_shape):
    x_size, y_size = new_shape
    image = transform.resize(image, (x_size, y_size),
                             preserve_range=True,
                             anti_aliasing=False, order=0).astype('uint8')

    return image


def plot_data(data, title):
    data = np.array(data)
    num_iters, rows, columns = data.shape
    # max value of seeds over surface
    max_val = np.max(data) + 1
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.10, bottom=0.25)
    val_init = num_iters // 2
    # Discrete color map with plt.cm.get_cmap()
    data_plt = plt.imshow(data[val_init], extent=[0, columns, 0, rows],
                          cmap=plt.cm.get_cmap('viridis', max_val), alpha=0.8, interpolation='nearest')
    plt.colorbar(ticks=range(max_val), label=title)
    plt.clim(-0.5, max_val-0.5)
    # plt.axis(aspect='image')
    plt.axis('off')

    ax_steps = plt.axes([0.20, 0.07, 0.70, 0.04])
    slider_steps = Slider(ax_steps, 'Steps', 0, num_iters - 1,
                          valinit=val_init, valstep=1)

    def update(val):
        step = slider_steps.val
        step = int(step)
        data_plt.set_data(data[step])

    slider_steps.on_changed(update)
    plt.show()
