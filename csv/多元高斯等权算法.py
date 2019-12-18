import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import multivariate_normal
import tools

# HT,PP,RP特征等权相乘
# ht+pp 10.10718%
# ht+rp 10.10653%
# ht+pp+rp 10.16%

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
all_data = np.loadtxt(data_source_file_name, dtype=float, delimiter=",", skiprows=1, usecols=tuple(range(3, 34)))

index1 = ht_index  # todo 考虑哪些特征
index2 = pp_index
index3 = rp_index
eer_array = []
for i in range(num_classes):
    train_data1 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass), index1]
    positive_test_data1 = all_data[
                          i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
                          index1]
    positive_scores1 = tools.calculate_score_by_cov(train_data1, positive_test_data1)
    train_data2 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass), index2]
    positive_test_data2 = all_data[
                          i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
                          index2]
    positive_scores2 = tools.calculate_score_by_cov(train_data2, positive_test_data2)
    # train_data3 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass), index3]
    # positive_test_data3 = all_data[
    #                       i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
    #                       index3]
    # positive_scores3 = tools.calculate_score_by_cov(train_data3, positive_test_data3)
    positive_scores = positive_scores1 * positive_scores2 # 用乘法来衔接两种特征
    negative_scores = []
    for j in range(num_classes):
        if j == i:
            continue
        negative_test_data1 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                              index1]
        negative_test_data2 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                              index2]
        # negative_test_data3 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
        #                       index3]
        negative_score1 = tools.calculate_score_by_cov(train_data1, negative_test_data1)
        negative_score2 = tools.calculate_score_by_cov(train_data2, negative_test_data2)
        # negative_score3 = tools.calculate_score_by_cov(train_data3, negative_test_data3)
        negative_scores += (negative_score1 * negative_score2).tolist()
    # negative_scores.sort()
    score_array = positive_scores.tolist() + negative_scores
    label_array = [1] * len(positive_scores) + [0] * len(negative_scores)
    far, tpr, thresholds = roc_curve(label_array, score_array)
    EER = tools.calculate_eer(far, tpr, thresholds)
    # print("user num: " + str(i) + ", EER: " + str(EER))
    eer_array.append(EER)
eer_array.sort()
print("eer=" + str(sum(eer_array) / len(eer_array)))
print("end")
