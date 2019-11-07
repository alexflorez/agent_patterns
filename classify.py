from simulation import simulation

import os
from collections import defaultdict
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
    feature_data = np.load("feature_data.npy")
    train_data = np.array(feature_data[:, 0: 400], dtype=np.float32)
    classes = feature_data[:, -1].tolist()
    score = classify(train_data, classes)
    print(f"Classification score: {score:.2f}")
