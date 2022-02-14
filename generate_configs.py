import configparser
from pathlib import Path
from itertools import product
from collections import namedtuple

BASE_PATH = Path("__file__").resolve().parent
CONFIG_BASE = BASE_PATH / "settings"

# Water parameters
WaterInitialLevel = 1
# WaterInitialEnergy = [10, 20, 30, 40, 50]
WaterInitialEnergy = [20, 30]
MoveLevel = 1
MoveEnergy = 1
Evaporate = [True, False]

# Plant parameters
PlantInitialLevel = 1
DeltaEnergy = 1
# PlantInitialEnergy = [5, 10, 15]
PlantInitialEnergy = [5]
# PointsHeight = [10, 20, 30, 40]
PointsHeight = [10, 20]
# GrowingEnergy = [5, 10, 15]
GrowingEnergy = [5]
# DropsToGrow = [3, 4, 5, 6, 7, 8]
DropsToGrow = [5]

# Experiment parameters
Dataset = "brodatz"
# Iterations = [100, 150, 200]
Iterations = [100]
# PercentagePlant = [20, 40, 60, 80, 100]
PercentagePlant = [20, 40, 60]
# PercentageWater = [50, 60, 70, 80, 90, 100]
PercentageWater = [50, 100]
# IterationsAddWater = [5, 10, 20, 30]
IterationsAddWater = [10, 20]

product_parameters = product(WaterInitialEnergy, Evaporate,
    PlantInitialEnergy, PointsHeight, GrowingEnergy, DropsToGrow,
    Iterations, PercentagePlant, PercentageWater, IterationsAddWater)

parameters = ["WaterInitialEnergy", "Evaporate",
              "PlantInitialEnergy", "PointsHeight", "GrowingEnergy", "DropsToGrow",
              "Iterations", "PercentagePlant", "PercentageWater", "IterationsAddWater"]
# parameters = ["WaterInitialLevel", "WaterInitialEnergy", "MoveLevel",
#               "MoveEnergy", "Evaporate",
#               "PlantInitialLevel", "DeltaEnergy", "PlantInitialEnergy",
#               "PointsHeight", "GrowingEnergy","DropsToGrow",
#               "Dataset", "Iterations", "PercentagePlant",
#               "PercentageWater", "IterationsAddWater"]
Parameters = namedtuple('Parameters', parameters)

for i, p in enumerate(product_parameters, 1):
    ps = Parameters(*p)
    config = configparser.ConfigParser()
    config.optionxform = str
    config["water"] = {"InitialLevel": WaterInitialLevel,
                       "InitialEnergy": ps.WaterInitialEnergy,
                       "MoveLevel": MoveLevel,
                       "MoveEnergy": MoveEnergy,
                       "Evaporate": ps.Evaporate}

    config["plant"] = {"InitialLevel": PlantInitialLevel,
                       "DeltaEnergy": DeltaEnergy,
                       "InitialEnergy": ps.PlantInitialEnergy,
                       "PointsHeight": ps.PointsHeight,
                       "GrowingEnergy": ps.GrowingEnergy,
                       "DropsToGrow": ps.DropsToGrow}

    config["experiment"] = {"Dataset": Dataset,
                            "Iterations": ps.Iterations,
                            "PercentagePlant": ps.PercentagePlant,
                            "PercentageWater": ps.PercentageWater,
                            "IterationsAddWater": ps.IterationsAddWater}

    configfile = CONFIG_BASE / f"confexp{i}.ini"
    with open(configfile, 'w') as cf:
        config.write(cf)
