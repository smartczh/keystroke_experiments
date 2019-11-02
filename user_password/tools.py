import os
import pickle

root_path = "C:/Users/Administrator/Desktop/击键特性相关论文/数据集/webkeystroke/output_numpy/passwords/"
userNum = 30  # 拿多少用户测试，一共有75个,todo 改动的话要把文件删了
allData_fileName = "./dataSource/all_data.pk"


# 读数据文件
def read_user_files(sample_type, feature_type, user, user_path, capture_line, all_data):  # sample_type:genuine和impostor, feature_type:pr和pp
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


def old_read_data():
    all_data = {}  # all_data[userid]["genuine/imposter"]["pr/pp"]=[[同一维的所有数据而不是不同维的一次输入数据],[],[]]
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
            if os.path.exists(captures_path):  # 被我手动过滤了几个user10，18（正样本太少)，而且他妈的009genuine是空的草SB todo 通过capture内容过滤好
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
                        read_user_files("genuine", "pr", user, user_path, capture_line, all_data)
                        read_user_files("genuine", "pp", user, user_path, capture_line, all_data)
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
    return all_data