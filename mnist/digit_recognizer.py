#!/usr/bin/env python3

from matplotlib import pyplot as plot
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances
from collections import Counter
import pandas as pd
import random as rn

TRAINING_SIZE = 42000

gridify_digit = lambda df, index: np.split(np.asarray(df.ix[index, 1:]), 28)
vectorize_digit = lambda df, index: np.asarray(df.ix[index, 1:])


def display_digit(grid, p=plot):
    p.imshow(grid, cmap="gray_r")

    p.tick_params(
        axis='y',
        which='both',
        left='off',
        right='off',
        labelleft='off',
        labelright='off')

    p.tick_params(
        axis='x',
        which='both',
        top='off',
        bottom='off',
        labeltop='off',
        labelbottom='off')

    p.show()


def prior_probability(df):
    labels = np.asarray(df.ix[:, 0])
    plot.figure()
    bins = np.arange(0, 11, 1)
    plot.hist(labels, bins=bins, normed=True, align='left')
    plot.xticks(range(0, 10, 1))
    plot.plot((-1, 10), (.1, .1), '--')
    plot.savefig("../images/distrib.png")


def one_of_each_digit(df, out_transx=gridify_digit):
    """
    Randomly picks one of each digit, generates a grid of
    it's pixels and returns a map from digit to pixels.

    :param df: a dataframe containting the full dataset
    :param out_transx: formats the pixels in the resultant
    map. Should be a function: (dataframe,index) → numpy
    array. By default this is `gridify_digits`

    :return: a map containing the digits 0-9 as the keys and
    a values as a tuple: (the pixels, the `id` the
    image from the dataset; i.e. 1 - 42000))
    """
    digits = dict(map(lambda i: (i, None), range(0, 10)))

    while not all(digits.values()):

        idx = rn.randrange(TRAINING_SIZE)
        digit = df.ix[idx, 0]

        if not digits[digit]:
            digits.update({digit: (out_transx(df, idx), idx)})

    return digits


def save_one_grid_of_each_digit(df):
    random_digits = one_of_each_digit(df)

    fig = plot.figure()
    fig.suptitle("Grid of Each Digit, Randomly Selected")

    for d in random_digits:
        p = fig.add_subplot(2, 5, d + 1)
        display_digit(p, random_digits[d][0])
        p.title.set_text("#%d\nDigit=%d" % (random_digits[d][1], d))

    fig.savefig("../images/one_grid_x_each_digit.png")


def plot_one_vec_of_each_digit(df):
    random_digits = one_of_each_digit(df, vectorize_digit)

    fig = plot.figure()
    fig.suptitle("Vector of Each Digit, Randomly Selected")

    data = np.asarray(list(map(lambda t: t[0], random_digits.values())))

    plot.imshow(data, cmap="gray_r", aspect=20)
    plot.yticks(range(0, 10, 1))
    plot.show()


def l2_distances(input_digits, compare_digits):
    distance = euclidean_distances(input_digits, compare_digits)
    distance = distance[np.tril(distance) != 0]
    return distance


def l2_match(input_digits, compare_digits):
    # a vector to record the matching minima
    minima = []

    for label, data in input_digits.items():
        min_vector = np.array([])
        min_distance = 1000000
        digit_tags = 0
        match_index = 0
        for dpoint in compare_digits:
            current_l2 = euclidean(data[0], dpoint[0])
            if (current_l2 < min_distance) and (current_l2 != 0.0):
                min_vector = dpoint[0]
                min_distance = current_l2
                digit_tag = dpoint[1]
                match_index = dpoint[2]
        minima.append([min_vector, min_distance, digit_tag, match_index])
    return minima


def plot_nearest_neighbor_for_each(df):
    # plot_nearest_neighbor_for_each(df0)
    random_digits = one_of_each_digit(df, vectorize_digit)
    allpts = [[df.ix[index, 1:], df.ix[index, 0], index] for index in range(df.shape[0])]
    minima = l2_match(random_digits, allpts)
    fig = plot.figure()
    fig.suptitle("Grid of Each Digit With Minimum L2 Match")
    for x in range(len(random_digits)):
        original_d = np.split(random_digits[x][0], 28)
        original_l = x
        match_d = np.split(minima[x][0], 28)
        match_l = minima[x][2]
        p = fig.add_subplot(4, 5, x + 1)
        display_digit(p, original_d)
        p.title.set_text("Digit=%d" % (original_l))
        p = fig.add_subplot(4, 5, x + 11)
        display_digit(p, match_d)
        p.title.set_text("Match=%d" % (match_l))
    fig.savefig("../images/matches.png")


def all_of_digit(df, digit):
    return [pd.Series.tolist(df.ix[index, 1:])
            for index in range(df.shape[0]) if df.ix[index, 0] == digit]


def binary_comp(df):
    zeros = all_of_digit(df, 0)
    ones = all_of_digit(df, 1)
    zeros_vs_zeros = l2_distances(zeros, zeros)
    ones_vs_ones = l2_distances(ones, ones)
    zeros_vs_ones = l2_distances(zeros, ones)
    plot.figure()
    plot.title("True Positives(b) and True Negatives(r): Binary Comparison")
    zvz = plot.hist(np.array(zeros_vs_zeros), bins=100, normed=True,
                    alpha=0.5, color="b", range=(0, 4500))
    ovo = plot.hist(np.array(ones_vs_ones), bins=100, normed=True,
                    alpha=0.5, color="b", range=(0, 4500))
    zvo = plot.hist(np.array(zeros_vs_ones), bins=100, normed=True,
                    alpha=0.5, color="r", range=(0, 4500))
    plot.savefig("../images/binary_comp.png")
    return zvz, zvo, ovo


def make_ROC(df):
    zvz, zvo, ovo = binary_comp(df)
    eer = 0
    tp = 0
    fn = sum(zvz[0])
    fp = 0
    tn = sum(zvo[0])
    roc_points = [[tp, fn, fp, tn]]
    for tau in range(len(zvz[0])):
        adj = (zvz[0][tau] + ovo[0][tau]) / 2
        tp += adj
        fn -= adj
        roc_points.append([tp, fn, fp, tn])
    for tau in range(len(zvo[0])):
        adj = zvo[0][tau]
        fp += adj
        tn -= adj
        current = roc_points[tau]
        current[2] = fp
        current[3] = tn
        roc_points[tau] = current
    tpr = []
    tnr = []
    eer_min = 100
    eer = []
    for value in range(len(roc_points[:-1])):
        c_tpr = roc_points[value][0] / (roc_points[value][0] + roc_points[value][1])
        c_tnr = roc_points[value][3] / (roc_points[value][2] + roc_points[value][3])
        tpr.append(c_tpr)
        tnr.append(c_tnr)
        if(abs(c_tpr - c_tnr) < eer_min and abs(c_tpr - c_tnr) != 0.0):
            eer_min = abs(c_tpr - c_tnr)
            eer = [c_tnr, c_tpr]
    plot.figure()
    plot.title("ROC")
    plot.ylim([0, 1])
    plot.ylabel("TPR")
    plot.xlim([0, 1])
    plot.xlabel("TNR")
    plot.text(eer[0], eer[1], str(eer[0])[:4] + "," + str(eer[1])[:4])
    plot.plot(eer[0], eer[1], "o")
    plot.plot(tnr, tpr, "-")
    plot.savefig("../images/ROC.png")


def kaggle_knn(train_df, test_df):
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


def make_bin(df, n, binn):
    return range(int((binn - 1) * (df.shape[0] / n)), int(binn * (df.shape[0] / n)))


def cross_validate(train_df):
    n = 3

    training_data = [np.asarray([pd.Series.as_matrix(train_df.ix[index, 1:])
                                 for index in make_bin(train_df, n, binn)]) for binn in range(1, n + 1)]

    training_labels = [np.asarray([train_df.ix[index, 0]
                                   for index in make_bin(train_df, n, binn)]) for binn in range(1, n + 1)]

    confusion_matrix = np.zeros((10, 10))
    for actual in range(10):
        confusion_matrix[actual][actual] = train_df.shape[0] * ((n * 2) / n)

    total_error = 0
    for test in range(n):
        for train_bin in range(n):
            if train_bin != test:
                classification = knn(training_data[train_bin], training_labels[
                                     train_bin], training_data[test])
                test_actual = training_labels[test]
                for x in range(len(classification)):
                    if classification[x][1] != test_actual[x]:
                        confusion_matrix[test_actual[x]][classification[x][1]] += 1
                        confusion_matrix[test_actual[x]][test_actual[x]] -= 1
                        total_error += 1

    percent_calc = np.full((10, 10), train_df.shape[0] * ((n * 2) / n))

    confusion_matrix = np.multiply(np.divide(confusion_matrix, percent_calc), 100)

    accuracy = 1 - (total_error / (train_df.shape[0] * ((n * 2) / n)))

    print(confusion_matrix)
    print(accuracy)

if __name__ == '__main__':
    df0 = pd.read_csv("kaggle_mnist_dataset/train.csv")
    df1 = pd.read_csv("kaggle_mnist_dataset/test.csv")
    # display_digit(gridify_digit(df0, 8))
    # prior_probability(df0)
    # plot_nearest_neighbor_for_each(df0)
    # binary_comp(df0)
    # make_ROC(df0)
    # kaggle_knn(df0, df1)
    # cross_validate(df0)
