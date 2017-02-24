#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.colors
import matplotlib as mpl
import matplotlib.pyplot as plt
PATH = "8.iris.data"


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print tip + '正确率：', np.mean(acc)


def get_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=',',converters={4: iris_type})
    x_data, y_data = np.split(data, (4,), axis=1)
    x_data = x_data[:, :2]
    return x_data, y_data

if __name__ == "__main__":
    x, y = get_data(PATH)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, random_state=1, train_size=0.6)
    # 分类器
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf = svm.SVC(C=0.1, kernel='rbf', decision_function_shape='ovr')
    print clf
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    # 准确率
    print "accuray:", clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    show_accuracy(y_hat, y_train, '训练集')
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    show_accuracy(y_hat, y_test, '测试集')