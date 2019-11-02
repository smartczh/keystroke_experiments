import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import tools
from scipy import stats

all_data = tools.old_read_data()  # all_data[userid]["genuine/imposter"]["pr/pp"][第几维]=[同一维的所有数据而不是不同维的一次输入数据]
print("load all_data success.")


# 记录表现比较好的user：41,6,28
for user in all_data.keys():
    if int(user) < 6 or len(all_data[user]["genuine"]["pr"][0])<200:  # todo 看数据用
        continue
    for data_list in all_data[user]["genuine"]["pr"]:
        filter_data_list = [x for x in data_list if 0.01 <= x <= 2]  # todo 过滤极值
        temp = np.array(filter_data_list)
        mean = temp.mean()
        std = temp.std()
        x = temp[(mean - 3 * std < temp) & (temp < mean + 3 * std)]  # todo 利用向量化计算和索引过滤偏差较大值
        if len(x) < 5:  # todo 数量太少
            print("pr, filter user:" + user)
            continue
        mean = x.mean()
        std = x.std()
        n, bins, patches = plt.hist(x, bins=40, color='r', alpha=0.5, density=True)  # 直方图
        plt.xlabel('time/s')
        plt.ylabel('norm frequency')
        # plt.title('histogram')
        y = stats.norm.pdf(bins, mean, std)
        plt.plot(bins, y, color='g', linewidth=1)
        print("pr, user:" + user+",data size:"+str(x.size)+",bin width:" + str(bins[1]-bins[0]))
        plt.show()
    for data_list in all_data[user]["genuine"]["pp"]:
        filter_data_list = [x for x in data_list if 0.01 <= x <= 2]  # todo 过滤极值
        temp = np.array(filter_data_list)
        mean = temp.mean()
        std = temp.std()
        x = temp[(mean - 3 * std < temp) & (temp < mean + 3 * std)]  # todo 利用向量化计算和索引过滤偏差较大值
        if len(x) < 5:  # todo 数量太少
            print("pp, filter user:" + user)
            continue
        mean = x.mean()
        std = x.std()
        n, bins, patches = plt.hist(x, bins=40, color='r', alpha=0.5, density=True)  # 直方图
        plt.xlabel('time/s')
        plt.ylabel('norm frequency')
        # plt.title('histogram')
        y = stats.norm.pdf(bins, mean, std)
        plt.plot(bins, y, color='g', linewidth=1)
        print("pp, user:" + user + ",data size:" + str(x.size) + ",bin width:" + str(bins[1] - bins[0]))
        plt.show()



