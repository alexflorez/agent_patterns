from simulation import simulation

import os
from collections import defaultdict
import numpy as np
import multiprocessing


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
            classes.append(class_.name)

    dataset = defaultdict(list)
    for cls_ in classes:
        with os.scandir(os.path.join(path_base, cls_)) as dataclass:
            for img in dataclass:
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
    feats = {"sum": data.sum(axis=(1, 2)),
             "max": data.max(axis=(1, 2)),
             "min": data.min(axis=(1, 2)),
             "count": np.count_nonzero(data, axis=(1, 2)),
             "mean_nz": mean_nz(data),
             "mean": data.mean(axis=(1, 2))}
    return feats[measure].tolist()


def extract_features(data_feats):
    """
    Extract feature vector
    data_feats correspond to 300 x n x m 
    """

    plant_data = data_feats[: 100, :, :]
    water_data = data_feats[100: 200, :, :]
    energy_data = data_feats[200: 300, :, :]

    feats_plant_nz = features(plant_data, "mean_nz")
    feats_water_nz = features(water_data, "mean_nz")
    feats_mass_plant = features(plant_data, "sum")
    feats_energy = features(energy_data, "sum")
    feats_count_plant = features(plant_data, "count")
    feats_plant = features(plant_data, "mean")

    feats = feats_plant_nz + feats_water_nz + feats_mass_plant + \
            feats_energy + feats_count_plant + feats_plant
    return feats


def collect_data(name_class, file_sample, num_iters):
    """
    Generate and concatenate data.
    """
    plant_data, water_data, energy_data = simulation(file_sample, num_iters)
    data = np.concatenate((plant_data, water_data, energy_data))
    data_cls = [data, name_class]
    return data_cls


if __name__ == '__main__':
    path_base = "bases/brodatz"
    class_samples = read_data(path_base)

    num_iters = 100
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # Adapt data to match arguments of pool.starmap
    class_samples_iters = [(cls_, smp, num_iters) for cls_, smp in class_samples]
    result_data = pool.starmap(collect_data, class_samples_iters)
    # Store the data
    np.save("data.npy", result_data)
    