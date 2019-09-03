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

    def region_idxs(self, x, y):
        """
        Extract a 3x3 region from the surface level 
        according to the current x and y positions.
        Return the indexes of the region.
        """
        rows, columns = self.level.shape
        ixs = []
        jys = []
        for i, j in product(range(-1, 2), range(-1, 2)):
            ix = (x + i + rows) % rows
            jy = (y + j + columns) % columns
            ixs.append(ix)
            jys.append(jy)
        return ixs, jys

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.level!r}>'
