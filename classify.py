from simulation import simulation

import os
from collections import defaultdict
import numpy as np
import pandas as pd
from skimage import io
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


def features(data):
    """
    Extract a feature vector of data. 
    """
    num_iters, rows, columns = data.shape
    vector = []
    for i in range(num_iters):
        x, y = data[i].nonzero()
        z = data[i][x, y]
        vector.append(z.mean())
    return vector


def classify(data):
    """
    Perform a classification using k-Nearest Neighbors 
    with 10-fold cross-validation scheme
    """
    classifier = KNeighborsClassifier(n_neighbors=1)
    # kfold: not greater than the number of members in each class
    kfold = 5
    train_data = data[data.columns[1:-1]].values
    predicted = cross_val_predict(classifier, train_data, data['class'], cv=kfold)
    score = metrics.accuracy_score(data['class'], predicted)
    return score


def extract_features(name_class, file_sample, num_iters):
    """
    Generate data and extract feature vector.
    """
    plant_data = simulation(file_sample, num_iters)
    feats = features(plant_data)
    feats_cls = feats + [name_class]
    return feats_cls


if __name__ == '__main__':
    path_base = "bases/brodatz"
    class_samples = read_data(path_base)

    num_iters = 20
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # Adapt data to match arguments of pool.starmap
    class_samples_iters = [(cls_, smp, num_iters) for cls_, smp in class_samples]
    result_data = pool.starmap(extract_features, class_samples_iters)
        
    feature_data = pd.DataFrame(result_data, columns=np.arange(num_iters + 1))
    feature_data = feature_data.rename(columns = {num_iters: 'class'})
    score = classify(feature_data)
    print(f"Classification score: {score:.2f}")
