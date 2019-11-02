import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats

#2019.10.29

# 读数据文件
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
                all_data[user][sample_type][feature_type].append([int(line) / 1000])
        else:
            if username_length == len(all_data[user][sample_type][
                                          feature_type]) - password_length:  # 他妈的有个文件username长度正好等于正确的账户和密码长度和，被我删掉了
                for i in range(0, len(username_content)):
                    all_data[user][sample_type][feature_type][i].append(int(username_content[i]) / 1000)
        f.close()
    password_path = root_path + user_path + sample_type + "/" + capture_line + "/" + "p_" + feature_type + ".txt"
    with open(password_path) as f:
        password_content = f.readlines()
        if len(all_data[user][sample_type][feature_type]) == username_length:
            for line in password_content:
                all_data[user][sample_type][feature_type].append([int(line) / 1000])
        else:
            if len(password_content) == len(
                    all_data[user][sample_type][
                        feature_type]) - username_length:  # 他妈的有个文件长度不一致C:\Users\Administrator\Desktop\击键特性相关论文\数据集\webkeystroke\output_numpy\passwords\user_001\genuine\2010-10-18T11_46_09
                for i in range(0, len(password_content)):
                    all_data[user][sample_type][feature_type][i + username_length].append(
                        int(password_content[i]) / 1000)
        f.close()

root_path = "D:/My Paper/小论文相关/数据集/webkeystroke/output_numpy/passwords/"
userNum = 30  # 拿多少用户测试，一共有75个,todo 改动的话要把文件删了
all_data = {}  # all_data[userid]["genuine/imposter"]["pr/pp"]=[[],[],[]]
allData_fileName = "./dataSource/all_data.pk"
if not os.path.exists(allData_fileName):
    for number in range(1, userNum + 1):
        # 设置好每个用户的路径
        if number < 10:
            user_path = "user_00" + str(number) + "/"
        elif 10 <= number < 100:
            user_path = "user_0" + str(number) + "/"
        else:
            user_path = "user_" + str(number) + "/"
        user = user_path[5:8]
        captures_path = root_path + user_path + "genuine/captures.txt"
        if os.path.exists(captures_path):  # 被我手动过滤了几个user10，18（正样本太少），而且他妈的009genuine是空的草SB
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
mainData_fileName = "./dataSource/main_data.pk"
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


for user in all_data.keys():
    for data_list in all_data[user]["genuine"]["pr"]:
        filter_data_list = [x for x in data_list if x <= 2]  # todo 过滤偏差较大值调整
        if len(filter_data_list) < 5:  # todo 数量太少
            continue
        temp = np.array(filter_data_list)
        mean = temp.mean()
        std = temp.std()
        filter_data_list1 = temp.tolist()
        filter_data_list2 = [x for x in filter_data_list1 if
                             mean - 5 * std <= x <= mean + 5 * std]  # todo 过滤偏差较大值调整
        if len(filter_data_list2) < 5:  # todo 数量太少
            continue
        filter_data_list2.sort()
        x = np.array(filter_data_list2)
        mean1 = x.mean()
        std1 = x.std()
        n, bins, patches = plt.hist(x, bins=80, color='r', alpha=0.5, rwidth=0.9, normed=0)  # 直方图
        plt.xlabel('time/s')
        plt.ylabel('frequency')
        plt.title('histogram')
        y = stats.norm.pdf(bins, mean1, std1)
        plt.plot(bins, y, color='g', linewidth=1)
        plt.show()
    for data_list in all_data[user]["genuine"]["pp"]:
        filter_data_list = [x for x in data_list if x <= 2]  # todo 过滤偏差较大值调整
        if len(filter_data_list) < 5:  # todo 数量太少
            continue
        temp = np.array(filter_data_list)
        mean = temp.mean()
        std = temp.std()
        filter_data_list1 = temp.tolist()
        filter_data_list2 = [x for x in filter_data_list1 if
                             mean - 5 * std <= x <= mean + 5 * std]  # todo 过滤偏差较大值调整
        if len(filter_data_list2) < 5:  # todo 数量太少
            continue
        filter_data_list2.sort()
        x = np.array(filter_data_list2)
        mean1 = x.mean()
        std1 = x.std()
        n, bins, patches = plt.hist(x, bins=80, color='r', alpha=0.5, rwidth=0.9, normed=1)  # 直方图
        plt.xlabel('time/s')
        plt.ylabel('frequency')
        plt.title('histogram')
        y = stats.norm.pdf(bins, mean1, std1)
        plt.plot(bins, y, color='g', linewidth=1)
        plt.show()



