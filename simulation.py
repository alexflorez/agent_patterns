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
    sim_data = np.zeros((num_iters, rows, columns))
    for i in range(num_iters):
        for _ in range(5):
            water.move()
        sim_data[i] = plant.seeds
        plant.grow(qty_grow)
        if i % 10 == 0:
            water.add()

    # max value of seed
    max_val = sim_data[-1].max()
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.10, bottom=0.25)
    #plt.imshow(water.height, extent=[0, 10, 0, 10], 
    #           cmap='Blues')
    val_init = num_iters // 2
    obj = plt.imshow(sim_data[val_init], extent=[0, columns, 0, rows], 
               cmap='Blues')
    plt.clim(0, max_val)
    plt.colorbar()
    plt.axis(aspect='image')

    ax_steps = plt.axes([0.10, 0.07, 0.80, 0.04])
    slider_steps = Slider(ax_steps, 'Steps', 0.0, num_iters - 1, 
                          valinit=val_init, valstep=1)

    def update(val):
        step = slider_steps.val
        step = int(step)
        obj.set_data(sim_data[step])

    slider_steps.on_changed(update)
    plt.show()