import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pickle


# 该程序用来计算eer和aoc,调整影响因子测试，相似度采用log(1+高斯值)后的加法

# 读取csv数据
# file_name = "data.csv"
# all_data = {}
# with open(file_name) as f:
#     reader = csv.reader(f)
#     reader_iter = iter(reader)  # 第一行是表头元素
#     next(reader_iter)
#     for row in reader_iter:
#         float_row = [float(i) for i in row[3:]]
#         if row[0] in all_data:
#             for i in range(0, len(float_row)):
#                 all_data[row[0]][i].append(float_row[i])
#         else:
#             all_data[row[0]] = []
#             for i in range(0, len(float_row)):
#                 all_data[row[0]].append([float_row[i]])
#     f.close()
# # 用户profile生成和数据过滤
# main_data = {}  # 存储平均值和方差
# user_begin = 0  # 训练集开始点
# user_end = 200  # 训练集结束点
# filter_product = 3
# for key in all_data.keys():
#     if key not in main_data:
#         main_data[key] = {}
#         main_data[key]["mean"] = []
#         main_data[key]["std"] = []
#     for data_list in all_data[key]:
#         temp = np.array(data_list[user_begin:user_end])  # 取中间值
#         mean = temp.mean()
#         std = temp.std()
#         filter_data_list = temp.tolist()
#         filter_data_list1 = [x for x in filter_data_list if
#                              mean - filter_product * std < x < mean + filter_product * std]  # 过滤偏差较大值
#         x = np.array(filter_data_list1)
#         mean = x.mean()
#         std = x.std()
#         main_data[key]["mean"].append(mean)
#         main_data[key]["std"].append(std)
#
# negative_begin = 0
# negative_end = 5
# column_length = 31  # todo 27之后是return
# mod = 2
# anti_mod = 2
#
#
# # 相似度计算函数，log(1+x),norm不norm差不多,0.3aoc最大，0.32eer最小
# def calculate_score(row_index, a, b):
#     similarity = 0
#     for column in range(0, column_length):
#         if column % 3 == 0:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2)) / (math.sqrt(2 * math.pi) * local_std)
#             score = a * math.log(1 + score) / 11
#             similarity = similarity + score
#         if column % 3 == 1:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (2 * local_std ** 2)) / (
#                     math.sqrt(2 * math.pi) * local_std)
#             score = b * math.log(1 + score) / 10
#             similarity = similarity + score
#     return similarity
#
#
# # 加法norm
# def calculate_score1(row_index, a, b):
#     similarity = 0
#     for column in range(0, column_length):
#         if column % 3 == 0:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2))
#             similarity = similarity + score * a / 11
#         if column % 3 == 1:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2))
#             similarity = similarity + score * b / 10
#     return similarity
#
#
# # distance
# def calculate_score2(row_index, a, b):
#     distance = 1.2
#     accept_num = 0
#     for column in range(0, column_length):
#         local_mean = main_data[user]["mean"][column]
#         local_std = main_data[user]["std"][column]
#         if local_mean - distance * local_std < all_data[compare][column][row_index] < local_mean + distance * local_std:
#             if column % 3 == 0:  # 模式变化
#                 accept_num += 1 * a / 11
#             if column % 3 == 1:  # 模式变化
#                 accept_num += 1 * b / 10
#     similarity = accept_num
#     return similarity
#
#
# # sqrt
# def calculate_score3(row_index, a, b):
#     similarity = 0
#     for column in range(0, column_length):
#         if column % 3 == 0:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2)) / (math.sqrt(2 * math.pi) * local_std)
#             similarity = similarity + math.sqrt(score) * a / 11
#         if column % 3 == 1:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2)) / (math.sqrt(2 * math.pi) * local_std)
#             similarity = similarity + math.sqrt(score) * b / 10
#     return similarity
#
#
# # 加法 不norm
# def calculate_score4(row_index, a, b):
#     similarity = 0
#     for column in range(0, column_length):
#         if column % 3 == 0:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2)) / (math.sqrt(2 * math.pi) * local_std)
#             similarity = similarity + score * a / 11
#         if column % 3 == 1:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2)) / (math.sqrt(2 * math.pi) * local_std)
#             similarity = similarity + score * b / 10
#     return similarity
#
#
# # log(1+x), norm
# def calculate_score5(row_index, a, b):
#     similarity = 0
#     for column in range(0, column_length):
#         if column % 3 == 0:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (
#                     2 * local_std ** 2))
#             score = a * math.log(1 + score) / 11
#             similarity = similarity + score
#         if column % 3 == 1:  # 模式变化
#             local_mean = main_data[user]["mean"][column]
#             local_std = main_data[user]["std"][column]
#             score = np.exp(-(all_data[compare][column][row_index] - local_mean) ** 2 / (2 * local_std ** 2))
#             score = b * math.log(1 + score) / 10
#             similarity = similarity + score
#     return similarity


if not (os.path.exists("results/x,eer_y,aoc_y,log(1+x),ab1.pk") and os.path.exists(
        "results/x,eer_y,aoc_y,+norm,ab.pk") and os.path.exists(
    "results/x,eer_y,aoc_y,distance1,ab.pk") and os.path.exists(
    "results/x,eer_y,aoc_y,sqrt,ab.pk") and os.path.exists(
    "results/x,eer_y,aoc_y,+,ab1.pk") and os.path.exists(
    "results/x,eer_y,aoc_y,log(1+x)norm,ab.pk")):  # todo 有多个文件
    # 生成参数范围供测试
    ab_range = []
    x = []
    eer_y = []
    aoc_y = []
    for gap in range(0, 101):
        argument1 = 0 + gap * 0.01
        argument2 = 1 - argument1
        ab_range.append([argument1, argument2])
    # 测试输出结果主体部分
    for argument in ab_range:
        eer_array = []
        auc_array = []
        a = argument[0]
        b = argument[1]
        for user in main_data.keys():  # 遍历每个基准用户
            positive_score_array = []
            negative_score_array = []
            for compare in all_data.keys():  # 遍历每个sample
                if compare == user:  # 只有一个
                    for row in range(user_end, 400):
                        total_score = calculate_score2(row, a, b)
                        positive_score_array.append(total_score)
                else:
                    for row in range(negative_begin, negative_end):
                        total_score = calculate_score2(row, a, b)
                        negative_score_array.append(total_score)
            positive_num = len(positive_score_array)
            negative_num = len(negative_score_array)
            score_array = positive_score_array + negative_score_array
            label_array = [1] * positive_num + [0] * negative_num
            far, tpr, thresholds = roc_curve(label_array, score_array)
            frr = 1 - tpr
            eer = tools.calculate_eer(far, frr)
            eer_array.append(eer)
            auc_value = auc(far, tpr)
            auc_array.append(auc_value)
        print("eer=" + str(sum(eer_array) / len(eer_array)) + " a=" + str(argument[0]) + " b=" + str(
            argument[1]) + " auc=" + str(sum(auc_array) / len(auc_array)))
        x.append(a)
        eer_y.append(sum(eer_array) / len(eer_array))
        aoc_y.append(sum(auc_array) / len(auc_array))
    with open("results/x,eer_y,aoc_y,distance1,ab.pk", "wb") as f:
        pickle.dump(x, f)
        pickle.dump(eer_y, f)
        pickle.dump(aoc_y, f)
else:
    with open("results/x,eer_y,aoc_y,log(1+x),ab1.pk", "rb") as f:
        x = pickle.load(f)
        eer_y = pickle.load(f)
        aoc_y = pickle.load(f)
    with open("results/x,eer_y,aoc_y,+norm,ab.pk", "rb") as f:
        x1 = pickle.load(f)
        eer_y1 = pickle.load(f)
        aoc_y1 = pickle.load(f)
    with open("results/x,eer_y,aoc_y,+,ab1.pk", "rb") as f:
        x2 = pickle.load(f)
        eer_y2 = pickle.load(f)
        aoc_y2 = pickle.load(f)
    # with open("results/x,eer_y,aoc_y,log(1+x)norm,ab.pk", "rb") as f:
    #     x3 = pickle.load(f)
    #     eer_y3 = pickle.load(f)
    #     aoc_y3 = pickle.load(f)
    # with open("results/x,eer_y,aoc_y,sqrt,ab.pk", "rb") as f:
    #     x4 = pickle.load(f)
    #     eer_y4 = pickle.load(f)
    #     aoc_y4 = pickle.load(f)
    with open("results/x,eer_y,aoc_y,distance1,ab.pk", "rb") as f:
        x5 = pickle.load(f)
        eer_y5 = pickle.load(f)
        aoc_y5 = pickle.load(f)
    plt.plot(x5, eer_y5, color='black', label="distance fusion")
    plt.plot(x1, eer_y1, color='red', label="mean fusion")
    plt.plot(x, eer_y, color='green', label="our method")
    #plt.plot(x2, eer_y2, color='red', label="Summation")
    # plt.plot(x3, eer_y3, color='blue', label="logarithm norm operation")
    # plt.plot(x4, eer_y4, color='yellow', label="sqrt operation")
    plt.legend(loc='upper left')
    plt.xlabel('proportion of HT&HTP')
    plt.ylabel('EER')
    #plt.title('EER of different proportion of hold time and flight time')
    plt.show()

