# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:18:18 2017

@author: vijay
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter


def knn(training_digits, training_labels, testing_digits):
    results = []
    k = 5
    counter = 1
    for unclpt in testing_digits:
        distances = cdist(np.array([unclpt]), training_digits)[0]
        ind = np.argpartition(distances, k)[:k]
        k_nn = distances[ind]
        k_values = np.ndarray.tolist(training_labels[ind])
        classify = max(k_values, key=k_values.count)
        results.append([counter, classify])
        counter += 1
    return results

train_df = pd.read_csv("kaggle_mnist_dataset/train.csv")
n = 3

def make_bin(df, n, binn):
    return range(int((binn-1)*(df.shape[0] / n)), int(binn*(df.shape[0] / n)))

training_data = [np.asarray([pd.Series.as_matrix(train_df.ix[index, 1:])
                 for index in make_bin(train_df, n, binn)]) for binn in range(1, n+1)]

training_labels = [np.asarray([train_df.ix[index, 0] for index in make_bin(train_df, n, binn)]) for binn in range(1, n+1)]
    
confusion_matrix = np.zeros((10, 10))
for actual in range(10):
    confusion_matrix[actual][actual] = train_df.shape[0]*((n*2)/n)

total_error = 0
for test in range(n):
    for train_bin in range(n):
        if train_bin != test:
            classification = knn(training_data[train_bin], training_labels[train_bin], training_data[test])
            test_actual = training_labels[test]
            for x in range(len(classification)):
                if classification[x][1] != test_actual[x]:
                    confusion_matrix[test_actual[x]][classification[x][1]] += 1
                    confusion_matrix[test_actual[x]][test_actual[x]] -= 1
                    total_error += 1
                    
percent_calc = np.full((10,10), train_df.shape[0]*((n*2)/n))

confusion_matrix = np.multiply(np.divide(confusion_matrix, percent_calc), 100)

accuracy = 1-(total_error/(train_df.shape[0]*((n*2)/n)))

print(confusion_matrix)
print(accuracy)
            
            