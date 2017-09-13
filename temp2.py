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
    
#error_scores = []
#    
#for test in range(n):
#    error = []
#    for train_bin in range(n):
#        if train_bin != test:
#            classification = knn(training_data[train_bin], training_labels[train_bin], training_data[test])
#            test_actual = training_labels[test]
#            for x in range(len(classification)):
#                if classification[x][1] != test_actual[x]:
#                    error.append([test_actual[x], classification[x][1]])
#    error_scores.append(error)
#
#folderror = []
#
#for test in error_scores:
#    errordict = {}
#    numerrors = len(test)
#    pct_acc = 1 - (numerrors/(len(training_data)/n))
#    print("Percent Accuracy for Fold:", pct_acc)
#    for mismatch in test:
#        if mismatch[0] not in errordict.keys():
#            errordict[mismatch[0]] = [mismatch[1]]
#        else:
#            errordict[mismatch[0]].append(mismatch[1])
#    folderror.append(errordict)
#    confusion_matrix = {}
#    for key in errordict.keys():
#        confusion_matrix[key] = Counter(errordict[key])
#        confusion_sum = 0
#        for confusionkey in confusion_matrix[key].keys():
#            confusion_matrix[key][confusionkey] /= len(errordict[key])
#            confusion_sum += confusion_matrix[key][confusionkey]
#        confusion_matrix[key][key] = 1-confusion_sum

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

accuracy = total_error/(train_df.shape[0]*((n*2)/n))

print(confusion_matrix)
print(accuracy)
            
            