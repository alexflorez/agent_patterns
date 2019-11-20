from features import extract_features

from itertools import combinations
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


def classify(train_data, classes):
    """
    Perform a classification using k-Nearest Neighbors 
    with 10-fold cross-validation scheme
    """
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(train_data)
    classifier = KNeighborsClassifier(n_neighbors=1)
    # kfold: not greater than the number of members in each class
    kfold = 10
    predicted = cross_val_predict(classifier, scaled_data, classes, cv=kfold)
    score = metrics.accuracy_score(classes, predicted)
    return score

if __name__ == '__main__':

    # read data
    feature_data = np.load("data.npy", allow_pickle=True)
    labels = feature_data[:, 1].tolist()
    data_to_process = feature_data[:, 0]

    plant = data_to_process[0][: 100, :, :]
    # print(plant.shape)
    water = data_to_process[0][100: 200, :, :]
    # print(water.shape)
    energy = data_to_process[0][200: 300, :, :]
    # print(energy.shape)

    data_features = [extract_features(dt) for dt in data_to_process]
    train_data = np.array(data_features, dtype=np.float32)
    # print(train_data.shape)

    # 1: plant_nz
    # 2: water_nz 
    # 3: plant_mass 
    # 4: energy
    # 5: plant_count
    # 6: plant_mean
    values = [1, 2, 3, 4, 5, 6] 
    # features to consider in classification
    start = 0
    stop = 100
    # number of iterations
    limit = 100            

    for i in range(1, len(values) + 1):
        for c in combinations(values, i):
            idxs = []
            for k in c:
                begin = (k - 1) * limit + start
                end = (k - 1) * limit + stop
                idxs.extend(range(begin, end))
            processed_data = np.array(train_data[:, idxs], dtype=np.float32)
            score = classify(processed_data, labels)
            print(f"{c} Classification score: {score:.2f}")
