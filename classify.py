from simulation import simulation

import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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


def features(data, func):
    """
    Extract a feature vector of data. 
    data is of shape (num_iters, rows, columns)
    """
    feats = {"sum": data.sum(axis=(1, 2)),
             "max": data.max(axis=(1, 2)),
             "mean": data.sum(axis=(1, 2)) / np.count_nonzero(data, axis=(1, 2))}
    return feats[func].tolist()


def classify(data):
    """
    Perform a classification using k-Nearest Neighbors 
    with 10-fold cross-validation scheme
    """
    classifier = KNeighborsClassifier(n_neighbors=1)
    # kfold: not greater than the number of members in each class
    kfold = 10
    train_data = data[data.columns[1:-1]].values
    predicted = cross_val_predict(classifier, train_data, data['class'], cv=kfold)
    score = metrics.accuracy_score(data['class'], predicted)
    return score


def extract_features(name_class, file_sample, num_iters):
    """
    Generate data and extract feature vector.
    """
    plant_data, water_data, energy_data = simulation(file_sample, num_iters)
    feats_plant = features(plant_data, "mean")
    feats_water = features(water_data, "mean")
    feats_mass_plant = features(plant_data, "sum")
    feats_energy = features(energy_data, "sum")
    feats_cls = feats_plant + feats_water + feats_energy + feats_mass_plant + [name_class]
    return feats_cls


if __name__ == '__main__':
    path_base = "bases/brodatz"
    class_samples = read_data(path_base)

    num_iters = 100
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # Adapt data to match arguments of pool.starmap
    class_samples_iters = [(cls_, smp, num_iters) for cls_, smp in class_samples]
    result_data = pool.starmap(extract_features, class_samples_iters)
    
    # create a dataframe to hold the feature vectors
    columns = num_iters * 4
    feature_data = pd.DataFrame(result_data, columns=np.arange(columns + 1))
    feature_data = feature_data.rename(columns = {columns: 'class'})
    np.save("feature_data.npy", feature_data)
    
    score = classify(feature_data)
    print(f"Classification score: {score:.2f}")
