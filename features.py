import argparse
import h5py
import multiprocessing
import numpy as np
from pathlib import Path
from sklearn import preprocessing


def features(data, measure):
    """
    Extract a feature vector of data. 
    data is of shape (num_iters, rows, columns)
    """
    axis = (1, 2)
    feats = {"sum": data.sum(axis=axis),
             "max": data.max(axis=axis),
             "count": np.count_nonzero(data, axis=axis),
             "mean": data.mean(axis=axis),
             "hist": np.histogram(data)[0]
             }
    dt = feats[measure]
    dt = dt[np.newaxis, :]
    return preprocessing.normalize(dt)


def extract_features(datafile):
    with h5py.File(datafile, 'r') as f:
        water_dt = f["water_dt"][:]
        plant_dt = f["plant_dt"][:]
        energy_dt = f["energy_dt"][:]
        # To read an scalar [()]
        label = f["label"][()]

    plant_count = features(plant_dt, "count")
    water_count = features(water_dt, "count")
    energy_count = features(energy_dt, "count")
    plant_sum = features(plant_dt, "sum")
    water_sum = features(water_dt, "sum")
    energy_sum = features(energy_dt, "sum")
    plant_max = features(plant_dt, "max")
    water_max = features(water_dt, "max")
    energy_max = features(energy_dt, "max")
    plant_hist = features(plant_dt, "hist")
    water_hist = features(water_dt, "hist")
    energy_hist = features(energy_dt, "hist")

    feats = [plant_count, water_count, energy_count,
             plant_sum, water_sum, energy_sum,
             plant_max, water_max, energy_max,
             plant_hist, water_hist, energy_hist]
    return np.hstack(feats), label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Folder data agents")
    args = parser.parse_args()
    exp_dir = Path(args.config)

    ds_name = "brodatz"
    ds_result = Path("results") / ds_name / exp_dir
    files = [f for f in ds_result.iterdir()]

    cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpus) as pool:
        fts_lbs = pool.map(extract_features, files)

    features_dt, labels = zip(*fts_lbs)
    features_dt = np.vstack(features_dt)
    hdf5_filename = f"{ds_name}_{exp_dir}.hdf5"
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset("features", data=features_dt)
        f.create_dataset("labels", data=labels)
