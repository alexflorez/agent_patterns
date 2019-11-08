from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def classify(train_data, classes):
    """
    Perform a classification using k-Nearest Neighbors 
    with 10-fold cross-validation scheme
    """
    classifier = KNeighborsClassifier(n_neighbors=1)
    # kfold: not greater than the number of members in each class
    kfold = 10
    predicted = cross_val_predict(classifier, train_data, classes, cv=kfold)
    score = metrics.accuracy_score(classes, predicted)
    return score

if __name__ == '__main__':

    # read data
    feature_data = np.load("feature_data.npy", allow_pickle=True)
    train_data = np.array(feature_data[:, 0: 400], dtype=np.float32)
    classes = feature_data[:, -1].tolist()
    
    # 1: plant
    # 2: water 
    # 3: energy 
    # 4: mass 
    values = [1, 2, 3, 4]
    # features to consider in classification
    start = 20
    stop = 95
    # number of iterations
    limit = 100
    for i in values:
        for c in combinations(values, i):
            idxs = []
            for k in c:
                begin = (k - 1) * limit + start
                end = (k - 1) * limit + stop
                idxs.extend(range(begin, end))
            train_data = np.array(feature_data[:, idxs], dtype=np.float32)
            score = classify(train_data, classes)
            print(f"{c} Classification score: {score:.2f}")
            
