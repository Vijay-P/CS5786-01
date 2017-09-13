# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 06:46:59 2017

@author: vijay
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

with open("submission.csv", "w+") as f:
    f.write("ImageId,Label\n")

    train_df = pd.read_csv("kaggle_mnist_dataset/train.csv")
    test_df = pd.read_csv("kaggle_mnist_dataset/test.csv")

    training_digits = np.asarray([pd.Series.as_matrix(train_df.ix[index, 1:])
                                  for index in range(train_df.shape[0])])

    training_labels = np.asarray([train_df.ix[index, 0] for index in range(train_df.shape[0])])

    testing_digits = np.asarray([pd.Series.as_matrix(test_df.ix[index, 0:])
                                 for index in range(test_df.shape[0])])

    k = 5

    counter = 1
    for unclpt in testing_digits:
        distances = cdist(np.array([unclpt]), training_digits)[0]
        ind = np.argpartition(distances, k)[:k]
        k_nn = distances[ind]
        k_values = np.ndarray.tolist(training_labels[ind])
        classify = max(k_values, key=k_values.count)
        f.write(str(counter) + "," + str(classify) + "\n")
        counter += 1
