import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import multivariate_normal
import tools
import pickle


# 等权互乘
# ht+pp 10.10718%
# ht+rp 10.10653%
# ht+pp+rp 10.16%

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fuse_score1(score1, a, score2,
                b):  # 要向量化传入ndarray，9.74%（有负数），固赋权不合理,本质和fuse_score2是一样的。将量纲提升到毫秒级，所有分数都在[0,1)，Ln之后确保都是负数，赋权合理了。
    if score1 == 0 or score2 == 0:
        return 0
    if score1 >= 1 or score2 >= 1:
        print("impossible!")
    else:
        return sigmoid(a * np.log(score1) + b * np.log(score2))


def fuse_score2(score1, a, score2, b):  # 传入ndarray，0-1和1到无穷的状态不一致，固赋权也不合理
    return np.power(score1, a) * np.power(score2, b)


fuse_score_vec1 = np.vectorize(fuse_score1)

data_source_file_name = "./dataSource/data.csv"

num_classes = 51  # 51个志愿者
dataNum_eachClass = 400  # 每种样本的数据量
num_feature = 31  # 31维数据
train_ratio = 0.7
test_ratio = 1 - train_ratio
negative_begin = 0
negative_end = 5
ht_index = list(range(0, 30, 3))
pp_index = list(range(1, 31, 3))
rp_index = list(range(2, 31, 3))
all_data = np.loadtxt(data_source_file_name, dtype=float, delimiter=",", skiprows=1, usecols=tuple(range(3, 34))) * 1000 # 变成以毫秒为单位

index1 = ht_index  # todo 考虑哪些特征
index2 = pp_index
index1_size = len(index1)
index2_size = len(index2)
ab_range = []
x = []
eer_y = []
for gap in range(0, 101):
    argument1 = 0 + gap * 0.01
    argument2 = 1 - argument1
    ab_range.append([argument1, argument2])
for argument in ab_range:
    eer_array = []
    a = argument[0]
    b = argument[1]
    for i in range(num_classes):
        train_data1 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass),
                      index1]
        positive_test_data1 = all_data[
                              i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
                              index1]
        positive_scores1 = tools.calculate_score_by_cov(train_data1, positive_test_data1)
        train_data2 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass),
                      index2]
        positive_test_data2 = all_data[
                              i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
                              index2]
        positive_scores2 = tools.calculate_score_by_cov(train_data2, positive_test_data2)
        positive_scores = fuse_score_vec1(positive_scores1, a, positive_scores2, b)
        negative_scores = []
        for j in range(num_classes):
            if j == i:
                continue
            negative_test_data1 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                                  index1]
            negative_test_data2 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                                  index2]
            negative_score1 = tools.calculate_score_by_cov(train_data1, negative_test_data1)
            negative_score2 = tools.calculate_score_by_cov(train_data2, negative_test_data2)
            negative_scores += (fuse_score_vec1(negative_score1, a, negative_score2, b)).tolist()
        score_array = positive_scores.tolist() + negative_scores
        label_array = [1] * len(positive_scores) + [0] * len(negative_scores)
        far, tpr, thresholds = roc_curve(label_array, score_array)
        EER = tools.calculate_eer(far, tpr, thresholds)
        # print("user num: " + str(i) + ", EER: " + str(EER))
        eer_array.append(EER)
    print("eer=" + str(sum(eer_array) / len(eer_array)) + " a=" + str(argument[0]) + " b=" + str(
        argument[1]))
    x.append(a)
    eer_y.append(sum(eer_array) / len(eer_array))
with open("./results/x,eer_y,(sigmoid(alnx+blny)).pk", "wb") as f:
    pickle.dump(x, f)
    pickle.dump(eer_y, f)
