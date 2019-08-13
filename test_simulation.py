import numpy as np
from surface import Surface
from water import Water


def test_create_surface():
    filename = 'images/flat.jpg'
    surface = Surface(filename)
    assert surface.level.shape == (10, 10)

def test_create_water():
    filename = 'images/flat.jpg'
    surface = Surface(filename)
    water = Water(surface)
    assert np.all(water.height == 1)