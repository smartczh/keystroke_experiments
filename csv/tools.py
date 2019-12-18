import os
import pickle
import csv
import math
import numpy as np
from scipy.stats import multivariate_normal
from scipy import stats
import heapq

file_name = "./dataSource/data.csv"
allData_fileName = "./dataSource/all_data.pk"


def old_read_data():
    all_data = {}
    if not os.path.exists(allData_fileName):
        with open(file_name) as f:
            reader = csv.reader(f)
            reader_iter = iter(reader)  # 第一行是表头元素
            next(reader_iter)
            for row in reader_iter:
                float_row = [float(i) for i in row[3:]]
                if row[0] in all_data:
                    for i in range(0, len(float_row)):
                        all_data[row[0]][i].append(float_row[i])
                else:
                    all_data[row[0]] = []
                    for data in float_row:
                        all_data[row[0]].append([data])
        with open(allData_fileName, 'wb') as f:
            pickle.dump(all_data, f)
    else:
        with open(allData_fileName, 'rb') as f:
            all_data = pickle.load(f)
    return all_data


def gaussian(sigma, x, u):
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    return y


def calculate_eer(far, tpr, threshold):
    fnr = 1 - tpr
    # eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
    EER1 = far[np.nanargmin(np.absolute((fnr - far)))]
    EER2 = fnr[np.nanargmin(np.absolute((fnr - far)))]
    return min(EER1, EER2)


def filter_data(data, k):  # data是一个list或者ndarray（一维），过滤极值，并过滤偏离k个标准差的数值
    if isinstance(data, list):
        data = np.array(data)
    data = data[(0.01 < data) & (data < 2)]  # 过滤极值
    mean = data.mean()
    std = data.std()
    return data[(mean - k * std < data) & (data < mean + k * std)]  # 利用向量化计算和索引过滤偏差较大值


def calculate_score_by_cov(train_data, test_data):  # return 一个ndarray
    train_cov = np.cov(train_data, rowvar=False)  # 计算协方差矩阵
    mean = np.mean(train_data, axis=0, keepdims=True)[0]  # axis=1返回横轴平均值,0纵轴平均值
    return multivariate_normal.pdf(test_data, mean=mean, cov=train_cov)


def calculate_score_by_std(train_data, test_data):  # return 一个ndarray
    mean = np.mean(train_data, axis=0, keepdims=True)[0]  # axis=1返回横轴平均值,0纵轴平均值
    std = np.std(train_data, axis=0, keepdims=True)[0]  # axis=1返回横轴平均值,0纵轴平均值
    temp = stats.norm.pdf(test_data, mean, std)
    return temp.prod(axis=1, keepdims=True)


def calculate_score_by_Manhattan(train_data, test_data):  # return 一个ndarray
    mean = np.mean(train_data, axis=0, keepdims=True)[0]  # axis=1返回横轴平均值,0纵轴平均值
    temp = test_data - mean
    return np.sum(-np.abs(temp), axis=1, keepdims=True)


def select_eer(eer_list):
    n_largest = 3
    num_classes = 51  # 51个志愿者
    select = num_classes - n_largest
    eer_list.sort()
    return eer_list[0:select]
