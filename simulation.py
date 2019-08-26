import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from surface import Surface
from water import Water
from plant import Plant
plt.style.use('seaborn')


if __name__ == '__main__':
    filename = 'images/tinybird.jpg'
    surface = Surface(filename)
    surface.reduce_to(10)
    water = Water(surface)
    plant = Plant(surface, water)
    plant.seed(10)
    qty_grow = 5
    
    num_iters = 100
    rows, columns = surface.level.shape
    seed_data = np.zeros((num_iters, rows, columns))
    # water_data = np.zeros((num_iters, rows, columns))
    for i in range(num_iters):
        for _ in range(5):
            water.move()
        seed_data[i] = plant.seeds
        # water_data[i] = water.height
        plant.grow(qty_grow)
        if i % 10 == 0:
            water.add()

    filedata = "tinybird.npy"
    if not os.path.isfile(filedata):
        np.save(filedata, seed_data)
    # load data
    # seed_data = np.load(filedata)
    # num_iters, rows, columns = seed_data.shape

    # max value of seed
    max_val = seed_data[-1].max()
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.10, bottom=0.25)
    val_init = num_iters // 2
    # Discrete color map with plt.cm.get_cmap()
    # water_plt = plt.imshow(water_data[val_init], extent=[0, columns, 0, rows], 
    #                        cmap=plt.cm.get_cmap('Blues'), interpolation='nearest')
    seeds_plt = plt.imshow(seed_data[val_init], extent=[0, columns, 0, rows], 
                           cmap=plt.cm.get_cmap('Greens', max_val), alpha=0.8, interpolation='nearest')
    plt.clim(0, max_val)
    plt.colorbar(label='Number of seeds')
    plt.axis(aspect='image')
    plt.axis('off')

    ax_steps = plt.axes([0.20, 0.07, 0.70, 0.04])
    slider_steps = Slider(ax_steps, 'Steps', 0, num_iters - 1, 
                          valinit=val_init, valstep=1)

    def update(val):
        step = slider_steps.val
        step = int(step)
        # water_plt.set_data(water_data[step])
        seeds_plt.set_data(seed_data[step])

    slider_steps.on_changed(update)
    plt.show()
