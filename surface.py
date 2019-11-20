import functools
import numpy as np
from skimage import io


class Surface:
    def __init__(self, n_region):
        self.level = None    
        self.x_idxs, self.y_idxs = (None, None)
        self.n_region = n_region

    def from_file(self, filename):
        """
        Create a surface level by reading an image file.
        The image is converted into grayscale.
        """
        self.level = io.imread(filename, as_gray=True)
        self.level = self.level * 255
        self.level = self.level.astype(int)
        self.x_idxs, self.y_idxs = self.idxs_region()

    def from_data(self, filedata):
        """
        Create a surface level by reading numpy data. 
        The data is stored as numpy format ".npy".
        """
        self.level = np.load(filedata)
        self.x_idxs, self.y_idxs = self.idxs_region()

    def reduce_to(self, percentage):
        """
        Change the surface level according to the 
        specified percentage.
        """
        self.level = np.array(self.level * percentage // 100,
                              dtype=int)
    
    def idxs_region(self):
        rows, columns = self.level.shape
        m = self.n_region // 2
        xs = list(range(rows - m, rows)) + list(range(rows)) + list(range(m))
        ys = list(range(columns - m, columns)) + list(range(columns)) + list(range(m))
        x_idxs, y_idxs = np.meshgrid(xs, ys, indexing='ij')
        return x_idxs, y_idxs

    @functools.lru_cache(maxsize=None)
    def region_idxs(self, x, y):
        """
        Extract a n x n region from the surface level 
        according to the current x and y positions.
        Return the indexes of the region.
        """
        n = self.n_region
        ixs = self.x_idxs[x: x + n, x: x + n]
        jys = self.y_idxs[y: y + n, y: y + n]
        return ixs, jys

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.level!r}>'
