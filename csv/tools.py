import os
import pickle
import csv
import math
import numpy as np

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


def filter_data(data, k):
    if isinstance(data, list):
        data = np.array(data)
    data = data[(0.01 < data) & (data < 2)]  # 过滤极值
    mean = data.mean()
    std = data.std()
    return data[(mean - k * std < data) & (data < mean + k * std)]  # 利用向量化计算和索引过滤偏差较大值