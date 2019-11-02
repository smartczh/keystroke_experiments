import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tools
from scipy import stats

# 2019.11.01 该程序主要用来证明高斯分布
# 编程问题：如何从中开始循环去掉头尾等，循环如何改变源数据而非取出，如何转化list的每个数据，如何过滤

all_data = tools.old_read_data()  # all_data[userid][31维特征]=[同一维的所有数据]
print("load all_data success.")

# 看一些比较规律的数据直接作图用于论文
examples = [("s003", 21), ("s004", 6), ("s008", 24), ("s002", 22), ("s007", 25), ("s010", 1)]
for example in examples:
    key = example[0]
    column = example[1]
    data_list = all_data[key][column]
    x = tools.filter_data(data_list, 3)
    mean = x.mean()
    std = x.std()
    print("user=" + key + " column=" + str(column) + " mean=" + str(mean) + " std=" + str(std) + " first=" + str(
        data_list[0]))
    n, bins, patches = plt.hist(x, bins=80, color='r', alpha=0.5, density=True)  # 直方图alpha控制颜色透明度
    plt.xlabel('time/s')
    plt.ylabel('normed frequency')
    # plt.title('histogram')
    y = stats.norm.pdf(bins, mean, std)
    plt.plot(bins, y, color='g', linewidth=1)
    plt.show()

# #遍历看数据
# count = 0
# for key in all_data.keys():
#     # if count < 10:  # 提前看后面的人的数据
#     #     count += 1
#     #     continue
#     column = 0
#     for data_list in all_data[key]:
#         if column % 3 != 1:  # hold time 已经符合高斯分布,每个用户也不看太多数据
#             column += 1
#             continue
#         filter_data_list1 = data_list
#         # filter_data_list1 = [x for x in data_list if x < 1]#过滤极值
#
#         temp = np.array(filter_data_list1)
#         #temp = np.log(temp1)  # 数学适应
#         mean = temp.mean()
#         std = temp.std()
#         filter_data_list2 = temp.tolist()
#         filter_data_list3 = [x for x in filter_data_list2 if mean - 3 * std < x < mean + 3 * std]  # 过滤偏差较大值
#
#         x = np.array(filter_data_list3)
#         mean = x.mean()
#         std = x.std()
#         print("key=" + key + " column=" + str(column) + " mean=" + str(mean) + " std=" + str(std) + " first=" + str(
#             data_list[0]))
#         n, bins, patches = plt.hist(x, bins=80, color='r', alpha=0.5, rwidth=0.9, normed=1)  # 直方图
#         y = mlab.normpdf(bins, mean, std)
#         plt.plot(bins, y, color='g', linewidth=1)
#         plt.show()
#         column += 1
