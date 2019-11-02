import os
import pickle
import numpy as np
import scipy.stats
import math
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# 权重化的eer可能更有意义

def read_user_files(sample_type, feature_type):  # sample_type:genuine和impostor, feature_type:pr和pp
    username_path = root_path + user_path + sample_type + "/" + capture_line + "/" + "l_" + feature_type + ".txt"
    password_path = root_path + user_path + sample_type + "/" + capture_line + "/" + "p_" + feature_type + ".txt"
    with open(password_path) as f:
        password_content = f.readlines()
        password_length = len(password_content)
        f.close()
    with open(username_path) as f:
        username_content = f.readlines()
        username_length = len(username_content)
        # if user_path == "user_023/" and feature_type == "pp" and (password_length != 7 or username_length != 6):
        #     print("sailinmu!")
        if sample_type == "impostor":
            if username_length + password_length != len(all_data[user]["genuine"][feature_type]):
                return
        if len(all_data[user][sample_type][feature_type]) == 0:
            for line in username_content:
                if feature_type == "pr":
                    all_data[user][sample_type][feature_type].append([int(line) / 1000])
                else:
                    all_data[user][sample_type][feature_type].append([math.sqrt(int(line) / 1000)])
        else:
            if username_length == len(all_data[user][sample_type][
                                          feature_type]) - password_length:  # 他妈的有个文件username长度正好等于正确的账户和密码长度和，被我删掉了
                for i in range(0, len(username_content)):
                    if feature_type == "pr":
                        all_data[user][sample_type][feature_type][i].append(int(username_content[i]) / 1000)
                    else:
                        all_data[user][sample_type][feature_type][i].append(math.sqrt(int(username_content[i]) / 1000))
        f.close()
    password_path = root_path + user_path + sample_type + "/" + capture_line + "/" + "p_" + feature_type + ".txt"
    with open(password_path) as f:
        password_content = f.readlines()
        if len(all_data[user][sample_type][feature_type]) == username_length:
            for line in password_content:
                if feature_type == "pr":
                    all_data[user][sample_type][feature_type].append([int(line) / 1000])
                else:
                    all_data[user][sample_type][feature_type].append([math.sqrt(int(line) / 1000)])
        else:
            if len(password_content) == len(
                    all_data[user][sample_type][
                        feature_type]) - username_length:  # 他妈的有个文件长度不一致C:\Users\Administrator\Desktop\击键特性相关论文\数据集\webkeystroke\output_numpy\passwords\user_001\genuine\2010-10-18T11_46_09
                for i in range(0, len(password_content)):
                    if feature_type == "pr":
                        all_data[user][sample_type][feature_type][i + username_length].append(int(password_content[i]) / 1000)
                    else:
                        all_data[user][sample_type][feature_type][i + username_length].append(math.sqrt(abs(int(password_content[i]) / 1000)))
        f.close()


root_path = "C:/Users/Administrator/Desktop/击键特性相关论文/数据集/webkeystroke/output_numpy/passwords/"
userNum = 30  # 拿多少用户测试，一共有75,todo 改动的话要把文件删了
all_data = {}  # all_data[userid]["genuine/imposter"]["pr/pp"]=[[],[],[]]
allData_fileName = "all_data(sqrt).pk"
if not os.path.exists(allData_fileName):
    for number in range(1, userNum + 1):
        if number < 10:
            user_path = "user_00" + str(number) + "/"
        elif 10 <= number < 100:
            user_path = "user_0" + str(number) + "/"
        else:
            user_path = "user_" + str(number) + "/"
        user = user_path[5:8]
        captures_path = root_path + user_path + "genuine/captures.txt"
        if os.path.exists(captures_path):  # 被我过滤了几个user10，18，而且他妈的009genuine是空的草SB
            all_data[user] = {}
            all_data[user]["genuine"] = {}
            all_data[user]["genuine"]["pr"] = []
            all_data[user]["genuine"]["pp"] = []
            all_data[user]["impostor"] = {}
            all_data[user]["impostor"]["pr"] = []
            all_data[user]["impostor"]["pp"] = []
            with open(captures_path) as capture:
                capture_content = capture.readlines()
                for capture_line in capture_content:
                    capture_line = capture_line.rstrip('\n')
                    capture_line = capture_line.replace(":", "_")
                    read_user_files("genuine", "pr")
                    read_user_files("genuine", "pp")
                capture.close()
            captures_path = root_path + user_path + "impostor/captures.txt"
            with open(captures_path) as capture:
                capture_content = capture.readlines()
                for capture_line in capture_content:
                    capture_line = capture_line.rstrip('\n')
                    capture_line = capture_line.replace(":", "_")
                    read_user_files("impostor", "pr")
                    read_user_files("impostor", "pp")
    with open(allData_fileName, 'wb') as f:
        pickle.dump(all_data, f)
else:
    with open(allData_fileName, 'rb') as f:
        all_data = pickle.load(f)
print("load all_data success.")

main_data = {}  # 存储平均值和方差
train_begin = 0
train_rate = 0.5
filter_product = 5
mainData_fileName = "main_data(sqrt).pk"
if not os.path.exists(mainData_fileName):
    for user in all_data.keys():
        if user not in main_data:
            main_data[user] = {}
            main_data[user]["pr"] = {}
            main_data[user]["pp"] = {}
            main_data[user]["pr"]["mean"] = []
            main_data[user]["pr"]["std"] = []
            main_data[user]["pp"]["mean"] = []
            main_data[user]["pp"]["std"] = []
        for data_list in all_data[user]["genuine"]["pr"]:
            train_end = math.ceil(len(data_list) * train_rate)
            temp = np.array(data_list[train_begin:train_end])  # 取训练集
            filter_data_list = [x for x in temp if x <= 2]  # todo 过滤偏差较大值调整
            if len(filter_data_list) < 3:
                print("main_data too small " + str(user))
            temp = np.array(filter_data_list)
            mean = temp.mean()
            std = temp.std()
            filter_data_list1 = temp.tolist()
            filter_data_list2 = [x for x in filter_data_list1 if
                                 mean - filter_product * std <= x <= mean + filter_product * std]  # todo 过滤偏差较大值调整
            x = np.array(filter_data_list2)
            if len(filter_data_list2) < 3:
                print("main_data too small " + str(user))
            mean1 = x.mean()
            std1 = x.std()
            main_data[user]["pr"]["mean"].append(mean)
            main_data[user]["pr"]["std"].append(std)
        for data_list in all_data[user]["genuine"]["pp"]:
            train_end = math.floor(len(data_list) * train_rate)
            temp = np.array(data_list[train_begin:train_end])  # 取训练集
            filter_data_list = [x for x in temp if x <= 2]  # todo 过滤偏差较大值调整
            if len(filter_data_list) < 3:
                print("main_data too small " + str(user))
            temp = np.array(filter_data_list)
            mean = temp.mean()
            std = temp.std()
            filter_data_list1 = temp.tolist()
            filter_data_list2 = [x for x in filter_data_list1 if
                                 mean - filter_product * std <= x <= mean + filter_product * std]  # todo 过滤偏差较大值调整
            x = np.array(filter_data_list2)
            if len(filter_data_list2) < 3:
                print("main_data too small " + str(user))
            mean1 = x.mean()
            std1 = x.std()
            main_data[user]["pp"]["mean"].append(mean)
            main_data[user]["pp"]["std"].append(std)
    with open(mainData_fileName, 'wb') as f:
        pickle.dump(main_data, f)
else:
    with open(mainData_fileName, 'rb') as f:
        main_data = pickle.load(f)
print("load main_data success.")


# log(1+x)
def calculate_score(index, sample_type, a, b):
    similarity = 0
    for column in range(0, len(all_data[user][sample_type]["pr"])):
        value = all_data[user][sample_type]["pr"][column][index]
        local_mean = main_data[user]["pr"]["mean"][column]
        local_std = main_data[user]["pr"]["std"][column]
        score = np.exp(-(value - local_mean) ** 2 / (2 * local_std ** 2)) / (
                math.sqrt(2 * math.pi) * local_std)
        similarity = similarity + a * math.log(1 + score)
    for column in range(0, len(all_data[user][sample_type]["pp"])):
        value = all_data[user][sample_type]["pp"][column][index]
        local_mean = main_data[user]["pp"]["mean"][column]
        local_std = main_data[user]["pp"]["std"][column]
        score = np.exp(-(value - local_mean) ** 2 / (2 * local_std ** 2)) / (
                math.sqrt(2 * math.pi) * local_std)
        similarity = similarity + b * math.log(1 + score)
    return similarity


# norm加法
def calculate_score1(index, sample_type, a, b):
    similarity = 0
    for column in range(0, len(all_data[user][sample_type]["pr"])):
        value = all_data[user][sample_type]["pr"][column][index]
        local_mean = main_data[user]["pr"]["mean"][column]
        local_std = main_data[user]["pr"]["std"][column]
        score = np.exp(-(value - local_mean) ** 2 / (2 * local_std ** 2))
        similarity = similarity + a * score
    for column in range(0, len(all_data[user][sample_type]["pp"])):
        value = all_data[user][sample_type]["pp"][column][index]
        local_mean = main_data[user]["pp"]["mean"][column]
        local_std = main_data[user]["pp"]["std"][column]
        score = np.exp(-(value - local_mean) ** 2 / (2 * local_std ** 2))
        similarity = similarity + b * score
    return similarity


# 乘法不用分ab
def calculate_score2(index, sample_type, a, b):
    similarity = 1
    for column in range(0, len(all_data[user][sample_type]["pr"])):
        value = all_data[user][sample_type]["pr"][column][index]
        local_mean = main_data[user]["pr"]["mean"][column]
        local_std = main_data[user]["pr"]["std"][column]
        score = np.exp(-(value - local_mean) ** 2 / (2 * local_std ** 2)) / (
                math.sqrt(2 * math.pi) * local_std)
        similarity = similarity * a * score
    for column in range(0, len(all_data[user][sample_type]["pp"])):
        value = all_data[user][sample_type]["pp"][column][index]
        local_mean = main_data[user]["pp"]["mean"][column]
        local_std = main_data[user]["pp"]["std"][column]
        score = np.exp(-(value - local_mean) ** 2 / (2 * local_std ** 2)) / (
                math.sqrt(2 * math.pi) * local_std)
        similarity = similarity * b * score
    return similarity


# 置信区间范围
def calculate_score3(index, sample_type, a, b):
    distance = 1
    accept_num = 0
    temp = user
    for column in range(0, len(all_data[user][sample_type]["pr"])):
        value = all_data[user][sample_type]["pr"][column][index]
        local_mean = main_data[user]["pr"]["mean"][column]
        local_std = main_data[user]["pr"]["std"][column]
        if local_mean - distance * local_std < value < local_mean + distance * local_std:
            accept_num += 1 * a
    for column in range(0, len(all_data[user][sample_type]["pp"])):
        value = all_data[user][sample_type]["pp"][column][index]
        local_mean = main_data[user]["pp"]["mean"][column]
        local_std = main_data[user]["pp"]["std"][column]
        if local_mean - distance * local_std < value < local_mean + distance * local_std:
            accept_num += 1 * b
    similarity = accept_num
    return similarity


# 生成参数范围供测试
ab_range = []
for gap in range(0, 21):
    argument1 = 0 + gap * 0.05
    argument2 = 1 - argument1
    ab_range.append([argument1, argument2])
# 测试输出结果主体部分
for argument in ab_range:
    a = argument[0]
    b = argument[1]
    eer_array = []
    auc_array = []
    total_num = 0
    for user in main_data.keys():  # 遍历每个用户
        positive_score_array = []
        negative_score_array = []
        genuine_test_end = min(len(all_data[user]["genuine"]["pr"][0]),
                               len(all_data[user]["genuine"]["pp"][0]))  # todo 还没确认每个column长度相等，正常应该相等
        genuine_test_begin = math.ceil(genuine_test_end * train_rate) + 1
        if len(all_data[user]["genuine"]["pr"][0]) != len(all_data[user]["genuine"]["pp"][0]):
            print("genuine length not equal!user=" + str(user))
        for index in range(genuine_test_begin, genuine_test_end):
            total_score = calculate_score(index, "genuine", a, b)
            positive_score_array.append(total_score)
        impostor_test_end = min(len(all_data[user]["impostor"]["pr"][0]), len(all_data[user]["impostor"]["pp"][0]))
        if len(all_data[user]["impostor"]["pr"][0]) != len(all_data[user]["impostor"]["pp"][0]):
            print("impostor length not equal!user=" + str(user))
        for index in range(0, impostor_test_end):
            total_score = calculate_score(index, "impostor", a, b)
            negative_score_array.append(total_score)
        # 画图感受
        # up = max(negative_flag_array)
        # positive_array_for_plot = [x for x in positive_array if x < up]
        # plt.hist(negative_flag_array, bins=80, color='r', alpha=0.5, hold=1)  # 直方图,hold=1可以将图画在一起
        # plt.hist(positive_array_for_plot, bins=80, color='g', alpha=0.5, hold=1)  # 直方图
        # plt.show()
        positive_num = len(positive_score_array)
        negative_num = len(negative_score_array)
        pn_num = positive_num + negative_num
        total_num += pn_num
        positive_score_array.sort()
        negative_score_array.sort()
        score_array = positive_score_array + negative_score_array
        label_array = [1] * positive_num + [0] * negative_num
        fpr, tpr, thresholds = roc_curve(label_array, score_array)
        # plt.plot(fpr, tpr, color='darkorange')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.show()
        miss_rate = 1 - tpr
        temp_min = 10
        eer_index = 0
        for index in range(0, len(fpr)):
            if abs(miss_rate[index] - fpr[index]) < temp_min:
                eer_index = index
                temp_min = abs(miss_rate[index] - fpr[index])
        eer = (fpr[eer_index] + miss_rate[eer_index]) / 2
        eer_array.append(eer * pn_num)
        auc_value = auc(fpr, tpr)
        auc_array.append(auc_value * pn_num)
        # for pos_index in range(0, positive_num):
        #     fpr =/negative_num
        #     tpr = /positive_num
    print("eer=" + str(sum(eer_array) / total_num) + " a=" + str(argument[0]) + " b=" + str(
        argument[1]) + " auc=" + str(sum(auc_array) / total_num))  # 不需要再除以size
