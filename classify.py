from itertools import combinations
import argparse
import h5py

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def classify(features, classes, name_classifier):
    classifiers = {'kNN': KNeighborsClassifier(n_neighbors=3),
                   'LDA': LinearDiscriminantAnalysis(solver='lsqr', 
                                                     shrinkage='auto'),
                   'Gaussian': GaussianNB(),
                   'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                   'Logistic': LogisticRegression(solver='lbfgs', multi_class='auto', 
                                                  max_iter=5000, n_jobs=-1),
                   'RandomForest': RandomForestClassifier(n_estimators=100, 
                                                          max_depth=10,
                                                          n_jobs=-1)}
    classifier = classifiers[name_classifier]
    kfold = 10
    scores = cross_val_score(classifier, features, classes, cv=kfold)
    return scores.mean(), scores.std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify data')
    parser.add_argument('filedata',
                        metavar='datafile',
                        type=str,
                        help='file with feature data')

    # Execute the parse_args() method
    args = parser.parse_args()

    # load data to process hdf5 format
    datafile = args.filedata
    with h5py.File(datafile, 'r') as f:
        feats = f["features"][:]
        labels = f["labels"][:]

    # Features
    # 1. plant_count
    # 2. water_count
    # 3. energy_count
    # 4. plant_sum
    # 5. water_sum
    # 6. energy_sum
    # 7. plant_max
    # 8. water_max
    # 9. energy_max
    # features to use in classification
    values = [2, 5, 6]
    n_iters = 100
    # Indexes to use a sub range of features
    start = 0
    stop = n_iters
    limit = n_iters            

    len_previous_data = len(values) * n_iters
    n_bins = 10
    len_hist = 3 * n_bins
    for i in range(1, len(values) + 1):
        for c in combinations(values, i):
            idxs = []
            for k in c:
                begin = (k - 1) * limit + start
                end = (k - 1) * limit + stop
                idxs.extend(range(begin, end))
            len_idxs = len(idxs)
            idxs.extend(range(len_previous_data, len_previous_data + len_hist))
            train_data = feats[:, idxs]
            score_mean, score_std = classify(train_data, labels, "kNN")
            print(f"kNN {c} Classification: {score_mean:.2f} {score_std:.2f}")
            score_mean, score_std = classify(train_data, labels, "LDA")
            print(f"LDA {c} Classification: {score_mean:.2f} {score_std:.2f}")
            score_mean, score_std = classify(train_data, labels, "xgboost")
            print(f"XGBoost {c} Classification: {score_mean:.2f} {score_std:.2f}")
            score_mean, score_std = classify(train_data, labels, "Gaussian")
            print(f"Gaussian {c} Classification: {score_mean:.2f} {score_std:.2f}")
            score_mean, score_std = classify(train_data, labels, "Logistic")
            print(f"Logistic {c} Classification: {score_mean:.2f} {score_std:.2f}")
            score_mean, score_std = classify(train_data, labels, "RandomForest")
            print(f"RandomForest {c} Classification: {score_mean:.2f} {score_std:.2f}")
