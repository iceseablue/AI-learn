#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
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

def draw_2D_chart(x1, x2, y):
    x1_min, x1_max = x1.min(), x1.max()  # 第1维的范围
    x2_min, x2_max = x2.min(), x2.max()  # 第2维的范围
    x1_grid, x2_grid = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    # plt.plot(x1, x2)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
    plt.grid()
    plt.show()

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
    print '支撑向量：', clf.support_
    print x_train[clf.support_, :]
    # print x_train[clf.support_, 1]
    # joblib.dump(clf, "train_model.m")
    draw_2D_chart(x_train[:, 0], x_train[:, 1])
