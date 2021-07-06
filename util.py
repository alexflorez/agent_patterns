import numpy as np

from skimage import io
from skimage import transform
from skimage.util import img_as_ubyte
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from environment import shape_cross


def read(filename, resize=True):
    """ png files are in the range [0 255] np.uint8
        jpg files are in the range [0 1] np.float
    """
    image = io.imread(filename, as_gray=True)
    if image.dtype == np.float:
        image = img_as_ubyte(image)
    x_size, y_size = image.shape
    if resize:
        x_size, y_size = 100, 100
    image = transform.resize(image, (x_size, y_size),
                             preserve_range=True,
                             anti_aliasing=False, order=0).astype('uint8')

    return image


def groups(point, surface):
    """Starting from a point find a group of neighbors"""
    neighbors = {point}
    visited = deque()
    visited.append(point)
    while visited:
        xy = visited.pop()
        idxs = shape_cross(xy, surface.shape)
        ns = {ij for ij in idxs if surface[ij]}
        visited.extend(ns - neighbors)
        neighbors |= ns
    return neighbors


def plot_data(data):
    data = np.array(data)
    num_iters, rows, columns = data.shape
    # max value of seeds over surface
    max_val = data.max() + 1
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.10, bottom=0.25)
    val_init = num_iters // 2
    # Discrete color map with plt.cm.get_cmap()
    seeds_plt = plt.imshow(data[val_init], extent=[0, columns, 0, rows],
                           cmap=plt.cm.get_cmap('viridis', max_val), alpha=0.8, interpolation='nearest')
    plt.colorbar(ticks=range(max_val), label='Height of plants')
    plt.clim(-0.5, max_val+0.5)
    # plt.axis(aspect='image')
    plt.axis('off')

    ax_steps = plt.axes([0.20, 0.07, 0.70, 0.04])
    slider_steps = Slider(ax_steps, 'Steps', 0, num_iters - 1,
                          valinit=val_init, valstep=1)

    def update(val):
        step = slider_steps.val
        step = int(step)
        seeds_plt.set_data(data[step])

    slider_steps.on_changed(update)
    plt.show()
