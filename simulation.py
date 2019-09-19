from surface import Surface
from water import Water
from plant import Plant

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
plt.style.use('seaborn')


def simulation(filename, num_iters):
    """
    Perform a simulation of plant growth on a surface.
    """
    surface = Surface()
    surface.from_file(filename)
    water = Water(surface)
    water.add()
    plant = Plant(surface, water)
    plant_percentage = 10
    plant.seed(plant_percentage)

    # rules of simulation
    QTY_GROW = 5
    
    rows, columns = surface.level.shape
    plant_data = np.zeros((num_iters, rows, columns), dtype=np.uint8)
    water_data = np.zeros((num_iters, rows, columns), dtype=np.uint8)
    energy_data = np.zeros((num_iters, rows, columns), dtype=np.float32)
    for i in range(num_iters):
        water.move()
        plant.grow(QTY_GROW)
        plant_data[i] = plant.seeds
        water_data[i] = water.height
        energy_data[i] = plant.energy
        if i % 10 == 0:
            water.add()
    return plant_data, water_data, energy_data


if __name__ == '__main__':
    filename = 'images/c001_004.png'
    num_iters = 100
    # simulation data contains plant, water, and energy data
    plant_data, water_data, energy_data = simulation(filename, num_iters)
    """
    Visualize the plant growth
    """
    num_iters, rows, columns = plant_data.shape
    # max value of seeds over surface
    max_val = plant_data[-1].max() + 1
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.10, bottom=0.25)
    val_init = num_iters // 2
    # Discrete color map with plt.cm.get_cmap()
    seeds_plt = plt.imshow(plant_data[val_init], extent=[0, columns, 0, rows],
                           cmap=plt.cm.get_cmap('viridis', max_val), alpha=0.8, interpolation='nearest')
    plt.colorbar(ticks=range(max_val), label='Height of plants')
    plt.clim(-0.5, max_val-0.5)
    plt.axis(aspect='image')
    plt.axis('off')

    ax_steps = plt.axes([0.20, 0.07, 0.70, 0.04])
    slider_steps = Slider(ax_steps, 'Steps', 0, num_iters - 1, 
                          valinit=val_init, valstep=1)

    def update(val):
        step = slider_steps.val
        step = int(step)
        seeds_plt.set_data(plant_data[step])

    slider_steps.on_changed(update)
    plt.show()
