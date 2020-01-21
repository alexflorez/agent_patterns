import numpy as np
from surface import Surface
from water import Water
from plant import Plant
import pytest


@pytest.fixture
def create_surface():
    # filename = 'images/flat.jpg'
    filename = 'images/slope.pny'
    surface = Surface(filename, 3)
    return surface


@pytest.fixture
def create_water(create_surface):
    surface = create_surface
    water = Water(surface)
    water.add()
    return water


@pytest.fixture
def create_plant(create_water):
    water = create_water
    plant = Plant(water.surface, water)
    return plant


def test_check_shape_surface(create_surface):
    surface = create_surface
    assert surface.level.shape == (10, 10)


def test_height_water(create_water):
    water = create_water
    assert np.all(water.height == 1)


def test_add_water(create_water):
    water = create_water
    water.add(50)
    assert np.any(water.height == 2)


def test_move_water(create_water):
    water = create_water
    qty = water.height.sum()
    water.move()
    qty_after = water.height.sum()
    assert qty == qty_after


def test_plant_in_surface(create_plant):
    plant = create_plant
    plant.seed(10)
    assert plant.seeds.sum() == 10


def test_plant_grows(create_plant):
    plant = create_plant
    plant.seed(10)
    plant.water.move()
    plant.grow()
    assert np.any(plant.energy > 0.5)

