from simulation import simulation

import os
from pathlib import Path
from collections import defaultdict
from itertools import product
from itertools import repeat
import numpy as np
import multiprocessing


DIR = 'bases/models'
models_dir = Path(DIR)
models_dir.mkdir(parents=True, exist_ok=True)


def read_data(path_base):
    """
    Read the classes and samples of a dataset stored in path_base
    which has the following structure:
    dataset/
        class1/
            sample1
            sample2
            ...
        class2/
        ...
    """
    classes = []
    with os.scandir(path_base) as datapath:
        for class_ in datapath:
            if not class_.name.startswith('.'):
                classes.append(class_.name)

    dataset = defaultdict(list)
    for cls_ in classes:
        with os.scandir(os.path.join(path_base, cls_)) as dataclass:
            for img in dataclass:
                if not img.name.startswith('.'):
                    dataset[cls_].append(img.name)

    class_samples = []
    for cls_, smps in dataset.items():
        for smp in smps:
            filename = os.path.join(path_base, cls_, smp)
            class_samples.append((cls_, filename))
    return class_samples


def features(data, measure):
    """
    Extract a feature vector of data. 
    data is of shape (num_iters, rows, columns)
    """
    def mean_nz(data):
        count = np.count_nonzero(data, axis=(1, 2))
        count[count == 0] = 1
        return data.sum(axis=(1, 2)) / count

    feats = {
             "sum": data.sum(axis=(1, 2)),
             "max": data.max(axis=(1, 2)),
             "min": data.min(axis=(1, 2)),
             "count": np.count_nonzero(data, axis=(1, 2)),
             "mean_nz": mean_nz(data),
             "mean": data.mean(axis=(1, 2)),
              # number of bins by default is 10
             "hist": np.histogram(data)[0]
             }
    return feats[measure].tolist()


def extract_features(data_feats, n_iters, label):
    """
    Extract feature vector
    data_feats correspond to 3*n_iters x n x m 
    """
    n = n_iters
    plant_data = data_feats[: n, :, :]
    water_data = data_feats[n: 2 * n, :, :]
    energy_data = data_feats[2 * n: 3 * n, :, :]

    feats_plant_nz = features(plant_data, "mean_nz")
    feats_water_nz = features(water_data, "mean_nz")
    feats_mass_plant = features(plant_data, "sum")
    feats_energy = features(energy_data, "sum")
    feats_count_plant = features(plant_data, "count")
    feats_plant = features(plant_data, "mean")
    hist_plant = features(plant_data, "hist")
    hist_water = features(water_data, "hist")

    feats = feats_plant_nz + feats_water_nz + feats_mass_plant + \
            feats_energy + feats_count_plant + feats_plant + \
            hist_plant + hist_water
    return feats + [label]


def collect_data(*params):
    """
    Generate and concatenate data.
    """
    name_class, *rest_params = params
    plant_data, water_data, energy_data = simulation(rest_params)
    data = np.concatenate((plant_data, water_data, energy_data))
    filename, *rest = rest_params
    name = Path(filename).stem
    path = models_dir.joinpath(f"{name}.npy")
    np.save(path, data)


def store_data(pool, feature_data, params):
    # ni: num_iters
    # ta: times_add_water
    # tm: times_water_moves
    # pp: plant_percentage
    # format to save: ni_ta_tm_pp.npy
    ni, ta, tm, pp = params
    feature_data = np.array(feature_data)
    labels = feature_data[:, 1]
    data_to_process = feature_data[:, 0]

    data_feats = pool.starmap(extract_features,
                              zip(data_to_process, repeat(ni), labels))

    np.save(f"data_{ni}_{ta}_{tm}_{pp}.npy", data_feats)


if __name__ == '__main__':
    path_base = "bases/brodatz"
    cpus = multiprocessing.cpu_count()
    class_samples = read_data(path_base)
    num_iters = [100]
    times_add_water = [10]
    times_water_moves = [10]
    plant_percentage = [10]
    params = product(num_iters, times_add_water, times_water_moves, plant_percentage)
    with multiprocessing.Pool(processes=cpus, maxtasksperchild=16) as pool:
        for i, ps in enumerate(params):
            # Adapt data to match arguments of pool.starmap
            parameters = [(cls_, smp, *ps)
                          for cls_, smp in class_samples]
            pool.starmap(collect_data, parameters)
