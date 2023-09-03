import csv
import numpy as np


def sample_poisson_data_loader(path):
    csv_dir = path
    with open(csv_dir) as f:
        reader = csv.reader(f)
        X = [_ for _ in reader]
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = float(X[i][j])

    X = np.array(X)
    return X
