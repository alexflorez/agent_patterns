from features import extract_features
from features import store_data

from itertools import combinations
import argparse
import os
import sys
import numpy as np

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def classify(train_data, classes, name):
    """
    Perform a classification using k-Nearest Neighbors 
    with 10-fold cross-validation scheme
    """
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(train_data)

    classifiers = {'kNN': KNeighborsClassifier(n_neighbors=3),
                   'LDA': LinearDiscriminantAnalysis(solver='lsqr', 
                                                     shrinkage='auto'),
                   'Gaussian': GaussianNB(),
                   'Logistic': LogisticRegression(solver='lbfgs', multi_class='auto', 
                                                  max_iter=5000, n_jobs=-1),
                   'RandomForest': RandomForestClassifier(n_estimators=100, 
                                                          max_depth=10,
                                                          n_jobs=-1)}
    # kfold: not greater than the number of members in each class
    classifier = classifiers[name]
    kfold = 4
    scores = cross_val_score(classifier, scaled_data, classes, cv=kfold)
    return scores.mean(), scores.std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify data')
    parser.add_argument('filedata',
                        metavar='filedata',
                        type=str,
                        help='file with feature data')

    # Execute the parse_args() method
    args = parser.parse_args()

    # load data to process
    # data_ni_ta_tm_pp.npy
    # filedata = "data_ni_ta_tm_pp.npy"
    filedata = args.filedata

    if not os.path.isfile(filedata):
        print('The file does not exist')
        sys.exit()

    data = np.load(filedata, allow_pickle=True)
    labels = data[:, -1]
    labels = labels.tolist()
    feats_data = np.array(data[:, :-1], dtype=np.float32)
    
    # 1: plant_nz
    # 2: water_nz 
    # 3: plant_mass 
    # 4: energy
    # 5: plant_count
    # 6: plant_mean
    # 7: hist_plant
    # 8: hist_water
    values = [1, 2, 3, 4, 5, 6]
    # Getting the number of iterations
    n_iters = filedata.split("_")[1]
    n_iters = int(n_iters)
    # features to consider in classification
    start = 0
    stop = n_iters
    # number of iterations
    limit = n_iters            

    len_previous_data = len(values) * n_iters
    # 10 because of the number of bins
    len_hist = 2 * 10
    for i in range(1, len(values) + 1):
        for c in combinations(values, i):
            idxs = []
            for k in c:
                begin = (k - 1) * limit + start
                end = (k - 1) * limit + stop
                idxs.extend(range(begin, end))
            len_idxs = len(idxs)
            idxs.extend(range(len_previous_data, len_previous_data + len_hist))
            train_data = feats_data[:, idxs]
            # score_mean, score_std = classify(train_data, labels, "kNN")
            # print(f"kNN {c} Classification: {score_mean:.2f} {score_std:.2f}")
            score_mean, score_std = classify(train_data, labels, "LDA")
            print(f"LDA {c} Classification: {score_mean:.2f} {score_std:.2f}")
            #  score_mean, score_std = classify(train_data, labels, "Gaussian")
            #  print(f"Gaussian {c} Classification: {score_mean:.2f} {score_std:.2f}")
            #  score_mean, score_std = classify(train_data, labels, "Logistic")
            #  print(f"Logistic {c} Classification: {score_mean:.2f} {score_std:.2f}")
            #  score_mean, score_std = classify(train_data, labels, "RandomForest")
            #  print(f"RandomForest {c} Classification: {score_mean:.2f} {score_std:.2f}")
