import numpy as np
from skimage import io


class Surface:
    def __init__(self, filename):
        self.level = io.imread(filename, as_gray=True)
        self.level = self.level * 255
        self.level = self.level.astype(int)

    def reduce_to(self, percentage):
        self.level = np.array(self.level * percentage // 100,
                              dtype=int)

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.level!r}>'
