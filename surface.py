import array
import functools
from itertools import product
import numpy as np
from skimage import io


class Surface:
    def __init__(self):
        self.level = None        

    def from_file(self, filename):
        """
        Create a surface level by reading an image file.
        The image is converted into grayscale.
        """
        self.level = io.imread(filename, as_gray=True)
        self.level = self.level * 255
        self.level = self.level.astype(int)

    def from_data(self, filedata):
        """
        Create a surface level by reading numpy data. 
        The data is stored as numpy format ".npy".
        """
        self.level = np.load(filedata)

    def reduce_to(self, percentage):
        """
        Change the surface level according to the 
        specified percentage.
        """
        self.level = np.array(self.level * percentage // 100,
                              dtype=int)
    
    @functools.lru_cache(maxsize=None)
    def region_idxs(self, x, y, rows, columns):
        """
        Extract a n x n region from the surface level 
        according to the current x and y positions.
        Return the indexes of the region.
        """
        # rows, columns = self.level.shape
        # for a 3x3 region
        n = 3
        m = n // 2
        idxs = range(m - n + 1, n - m)
        size = range(n * n)
        ixs = array.array('B', size)
        jys = array.array('B', size)
        for k, (i, j) in enumerate(product(idxs, idxs)):
            ixs[k] = (x + i + rows) % rows
            jys[k] = (y + j + columns) % columns
        return ixs, jys

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.level!r}>'
