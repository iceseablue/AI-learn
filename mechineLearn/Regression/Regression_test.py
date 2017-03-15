#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import svm

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


def get_gyroscope_data():
    safe_path = '../data/gyroscope/safe_data.txt'  # 数据文件路径
    pre_alarm_path = '../data/gyroscope/pre_alarm_data.txt'  # 数据文件路径
    alarm_path = '../data/gyroscope/alarm_data.txt'  # 数据文件路径

    # # 路径，浮点型数据，逗号分隔，第4列使用函数iris_type单独处理
    safe_data_x = np.loadtxt(safe_path, dtype=float, delimiter="\t")
    safe_data_y = np.zeros(safe_data_x.shape[0])
    print "safe_data_x:", safe_data_x.shape
    print "safe_data_y:", safe_data_y.shape

    pre_alarm_data_x = np.loadtxt(pre_alarm_path, dtype=float, delimiter='\t')
    pre_alarm_data_y = np.zeros(pre_alarm_data_x.shape[0])
    pre_alarm_data_y[:] = 1

    print "pre_alarm_data_x:", pre_alarm_data_x.shape
    print "pre_alarm_data_y:", pre_alarm_data_y.shape

    alarm_data_x = np.loadtxt(alarm_path, dtype=float, delimiter='\t')
    alarm_data_y = np.zeros(alarm_data_x.shape[0])
    alarm_data_y[:] = 2

    print "alarm_data_x:", alarm_data_x.shape
    print "alarm_data_y:", alarm_data_y.shape

    x_data = np.vstack((safe_data_x, pre_alarm_data_x))
    x_data = np.vstack((x_data, alarm_data_x))
    y_data = np.append(safe_data_y, pre_alarm_data_y)
    y_data = np.append(y_data, alarm_data_y)

    print "x_data:", x_data.shape
    print "y_data:", y_data.shape

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1)
    print "x_train:", x_train.shape
    print "y_train:", y_train.shape
    print "x_test:", x_test.shape
    print "y_test:", y_test.shape
    return x_train, x_test, y_train, y_test

def get_iris_data():
    path = '../data/iris/8.iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x_data, y_data = np.split(data, (4,), axis=1)
    # x_data = x_data[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1, train_size=0.6)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    test_data_out_path = '../data/gyroscope/test_data.txt'

    # x_train, x_test, y_train, y_test = get_iris_data()
    x_train, x_test, y_train, y_test = get_gyroscope_data()
    # y_train =
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)

    lr = LogisticRegression()   # Logistic回归模型
    lr.fit(x_train, y_train.ravel())       # 根据数据[x,y]，计算回归参数

    y_hat = lr.predict(x_test)  # 预测值
    print "y_hat:", y_hat.shape
    print "y_hat_LR:", y_hat
    print "LR train score:", lr.score(x_train, y_train)
    print "LR test score:", lr.score(x_test, y_test)

    result = y_hat == y_test.ravel()
    result = np.array([1 if a == True else 0 for a in result])

    acc = np.mean(result)
    print "accaury:", float(result.sum())/result.shape[0]
    result_data = np.zeros((x_test.shape[0], x_test.shape[1]+2))
    result_data[:, :x_test.shape[1]] = x_test
    result_data[:, x_test.shape[1]] = y_test.ravel()
    result_data[:, x_test.shape[1]+1] = y_hat

    print "result_data:", result_data.shape
    fmat_str = '%f\t' * result_data.shape[1]
    np.savetxt(test_data_out_path, result_data, fmt=fmat_str, newline='\n')

    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf = svm.SVC(C=0.9, kernel='rbf', gamma=7, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel().T)
    print "SVM train: ", clf.score(x_train, y_train)  # 精度

    y_hat_svm = clf.predict(x_test)
    print "SVM test: ", clf.score(x_test, y_test)  # 精度
    print "y_hat_svm:\n", y_hat_svm



    # 等价形式
    # lr = Pipeline([('sc', StandardScaler()),
    #                     ('clf', LogisticRegression()) ])
    # lr.fit(x, y.ravel())

    # 画图
    # N, M = 500, 500     # 横纵各采样多少个值
    # x1_min, x1_max = x_train[:, 0].min(), x_train[:, 0].max()   # 第0列的范围
    # x2_min, x2_max = x_train[:, 1].min(), x_train[:, 1].max()   # 第1列的范围
    # t1 = np.linspace(x1_min, x1_max, N)
    # t2 = np.linspace(x2_min, x2_max, M)
    # x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    # x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

    # 无意义，只是为了凑另外两个维度
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    # cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    # cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    # y_hat = lr.predict(x_test)                  # 预测值
    # y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
    # plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
    # plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
    # plt.xlabel('petal length')
    # plt.ylabel('petal width')
    # plt.xlim(x1_min, x1_max)
    # plt.ylim(x2_min, x2_max)
    # plt.grid()
    # plt.savefig('2.png')
    # plt.show()

    # 训练集上的预测结果
    # y_hat = lr.predict(x)
    # y = y.reshape(-1)
    # result = y_hat == y
    # print y_hat
    # print result
    # acc = np.mean(result)
    # print '准确度: %.2f%%' % (100 * acc)
