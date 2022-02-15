import util
from water import Water
from plant import Plant
from environment import Environment
from datasource import DataSource

import configparser
from pathlib import Path
import h5py
from tqdm import tqdm
import multiprocessing
import argparse


def process_agent(file, label, configuration, destiny):
    confExperiment = configuration["experiment"]

    ITERATIONS = confExperiment.getint("Iterations")                    # 7
    PERCENT_PLANT = confExperiment.getint("PercentagePlant")            # 8
    PERCENT_WATER = confExperiment.getint("PercentageWater")            # 9
    ITERATIONS_ADD_WATER = confExperiment.getint("IterationsAddWater")  # 10
    EVAPORATE = configuration.getboolean("water", "Evaporate")          # 2

    confWater = configuration["water"]
    ENERGY_WATER = confWater.getint("InitialEnergy")                    # 1
    confPlant = config["plant"]
    ENERGY_PLANT = confPlant.getint("InitialEnergy")                    # 3
    POINTS_HEIGHT = confPlant.getint("PointsHeight")                    # 4
    GROW_ENERGY = confPlant.getint("GrowingEnergy")                     # 5
    DROPS_TO_GROW = confPlant.getint("DropsToGrow")                     # 6
    image = util.read(file)
    water = Water(image.shape, EVAPORATE)
    water.INIT_ENERGY = ENERGY_WATER
    water.add(PERCENT_WATER)
    plant = Plant(image.shape)
    plant.INIT_ENERGY = ENERGY_PLANT
    plant.POINTS_HEIGHT = POINTS_HEIGHT
    plant.GROW_ENERGY = GROW_ENERGY
    plant.DROPS_TO_GROW = DROPS_TO_GROW
    plant.add(PERCENT_PLANT)
    env = Environment(image)
    env.water = water

    water_dt = []
    plant_dt = []
    energy_dt = []
    # for i in tqdm(range(1, ITERATIONS + 1)):
    for i in range(1, ITERATIONS + 1):
        water_dt.append(water.level.copy())
        plant_dt.append(plant.level.copy())
        energy_dt.append(plant.energy.copy())
        water.move(env)
        plant.check_grow(env)
        if i % ITERATIONS_ADD_WATER == 0:
            # print(f"Iteration {i+1}")
            water.add(PERCENT_WATER)

    hdf5_filename = f"{destiny}/{file.stem}.hdf5"
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset("water_dt", data=water_dt)
        f.create_dataset("plant_dt", data=plant_dt)
        f.create_dataset("energy_dt", data=energy_dt)
        f.create_dataset("label", data=label)
    # return water_dt, plant_dt, energy_dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="configuration file")
    args = parser.parse_args()
    config_name = Path(args.config)

    config = configparser.ConfigParser()
    config.read(config_name)

    # Parameters of experiment
    confExperiment = config["experiment"]
    dataset = confExperiment["Dataset"]

    # Path to store the results of experiment
    ds_result = f"results/{dataset}/{config_name.stem}"
    cwd = Path.cwd()
    dir_result = cwd / ds_result
    if not dir_result.exists():
        Path.mkdir(dir_result, parents=True)
    #
    ds = DataSource(dataset)
    parameters = [(f, lb, config, ds_result) for f, lb in zip(ds.files, ds.labels)]
    cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpus) as pool:
        resp = pool.starmap(process_agent, parameters)

    # fl = ds.files[0]
    # lb = ds.labels[0]
    # process_agent(fl, lb, config, ds_result)
