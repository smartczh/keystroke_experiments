import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import multivariate_normal
import tools

data_source_file_name = "./dataSource/data.csv"

num_classes = 51  # 51个志愿者
dataNum_eachClass = 400  # 每种样本的数据量
num_feature = 31  # 31维数据
train_ratio = 0.7
test_ratio = 1 - train_ratio
negative_begin = 0
negative_end = 5
ht_index = list(range(0, 31, 3))
pp_index = list(range(1, 31, 3))
rp_index = list(range(2, 31, 3))
all_data = np.loadtxt(data_source_file_name, dtype=float, delimiter=",", skiprows=1, usecols=tuple(range(3, 34)))

index1 = ht_index  # todo 考虑哪些特征
index2 = pp_index
eer_array = []
for i in range(num_classes):
    train_data1 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass), index1]
    positive_test_data = all_data[
                         i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
                         index1]
    train_cov1 = np.cov(train_data, rowvar=False)  # 计算协方差矩阵
    mean = np.mean(train_data, axis=0, keepdims=True)[0]  # axis=1返回横轴平均值,0纵轴平均值
    positive_scores = multivariate_normal.pdf(positive_test_data, mean=mean, cov=train_cov).tolist()
    negative_scores = []
    for j in range(num_classes):
        if j == i:
            continue
        negative_test_data = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                             index]
        temp_scores = multivariate_normal.pdf(negative_test_data, mean=mean, cov=train_cov)
        negative_scores += temp_scores.tolist()
    # negative_scores.sort()
    score_array = positive_scores + negative_scores
    label_array = [1] * len(positive_scores) + [0] * len(negative_scores)
    far, tpr, thresholds = roc_curve(label_array, score_array)
    EER = tools.calculate_eer(far, tpr, thresholds)
    print("user num: " + str(i) + ", EER: " + str(EER))
    eer_array.append(EER)
eer_array.sort()
print("eer=" + str(sum(eer_array) / len(eer_array)))
