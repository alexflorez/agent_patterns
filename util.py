import numpy as np

from skimage import io
from skimage import transform
from skimage.util import img_as_ubyte
from collections import deque

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
