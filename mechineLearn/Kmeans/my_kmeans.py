#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.colors
import matplotlib as mpl
import matplotlib.pyplot as plt

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x_data, y_data = np.split(data, (4,), axis=1)
    x_data = x_data[:, :4]
    return x_data, y_data


def dist_calculate(x1, x2):
    # print "x1: ", x1
    # print "x2: ", x2
    dist = np.sqrt(np.sum(np.power(x1 - x2, 2)))
    # print "dist: ", dist
    return dist


def get_rand_center_array(data_set, k):
    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))

    for i in range(n):
        min_value = min(data_set[:, i])
        range_value = float(max(data_set[:, i])) - min_value
        centroids[:, i] = min_value + range_value*np.random.rand(k, 1)

    return centroids

def Kmeans(data_set, k, dist_meas=dist_calculate,
           create_center_matrix=get_rand_center_array):
    m = np.shape(data_set)[0]
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroids = create_center_matrix(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_meas(centroids[j, :].A[0], data_set[i, :])
                if min_dist > dist_ji:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
                cluster_assment[i, :] = min_index, min_dist**2
        for cent in range(k):
            pts_in_cluster = data_set[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pts_in_cluster, axis=0)
    return centroids, cluster_assment


def stastic_data(data_set, cluster_assment):
    for i in range(k):
        class_i = data_set[np.nonzero(cluster_assment[:, 0].A == i)[0]]
        print ">> ", i, ":\n"
        stastic = {}
        for j in class_i:
            key = j[0]
            if key in stastic.keys():
                stastic[key] += 1
            else:
                stastic[key] = 1
        if np.sum(stastic.values()) != 0:
            print "accuracy: ", \
                float(np.max(stastic.values()))*100 / np.sum(stastic.values()), \
                "%"


def predict_data(data_set, centroids, dist_meas=dist_calculate):
    min_index = -1
    min_dist = np.inf
    for i in range(np.shape(centroids)[0]):
        dist_value = dist_meas(data_set, centroids[:, i].A[0])
        if min_dist > dist_value:
            min_dist = dist_value
            min_index = i

    return min_index, centroids[min_index, :].A[0]


def draw_2D_chart(x_data, y_data):



if __name__ == "__main__":
    path = '../SVM/8.iris.data'
    k = 3
    x_data, y_data = load_data(path)
    centroids = get_rand_center_array(x_data, k)

    centroids, cluster_assment = Kmeans(x_data, k)
    print ">>\n", centroids
    print ">>\n", cluster_assment
    stastic_data(y_data, cluster_assment)

    # test_data = [8.0, 2.3, 3.3, 6.0]
    test_data = [5.0,2.3,3.3,1.0]

    print "predict result:\n",\
        predict_data(test_data, centroids)

