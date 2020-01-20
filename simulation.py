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
    surface = Surface(filename, n_region=3)
    water = Water(surface)
    water.add()
    plant = Plant(surface, water)
    plant_percentage = 10
    plant.seed(plant_percentage)

    # rules of simulation
    water_grow = 5
    times_add_water = 10
    times_water_moves = 10
    
    rows, columns = surface.level.shape
    plant_data = np.zeros((num_iters, rows, columns), dtype=np.uint8)
    water_data = np.zeros((num_iters, rows, columns), dtype=np.uint8)
    energy_data = np.zeros((num_iters, rows, columns), dtype=np.float32)
    for i in range(num_iters):
        plant_data[i] = plant.seeds
        water_data[i] = water.height
        energy_data[i] = plant.energy
        for _ in range(times_water_moves):
            water.move()
        plant.grow(water_grow)
        if i % times_add_water == 0:
            water.add()
    return plant_data, water_data, energy_data


def check_movement_water(filename, num_iters):
    surface = Surface(filename, n_region=3)
    water = Water(surface)
    data = np.zeros_like(surface.level)
    #data[5, 0] = 1
    data[:, 1] = 1
    water.set_data(data)
    #water.add()
    
    rows, columns = surface.level.shape
    water_data = np.zeros((num_iters, rows, columns), dtype=np.uint8)
    for i in range(num_iters):
        water_data[i] = water.height
        water.move()
    return water_data
    

def plot_data(data):
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
    plt.clim(-0.5, max_val-0.5)
    plt.axis(aspect='image')
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
        

if __name__ == '__main__':
    filename = 'images/c001_004.png'
    filename = 'images/slope.npy'
    num_iters = 30
    # simulation data contains plant, water, and energy data
    # plant_data, water_data, energy_data = simulation(filename, num_iters)
    water_data = check_movement_water(filename, num_iters)
    """
    Visualize the generated data
    """
    plotting = True
    if plotting:
        plot_data(water_data)
