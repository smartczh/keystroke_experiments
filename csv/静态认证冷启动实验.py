import numpy as np
import math
from sklearn.metrics import roc_curve, auc
from scipy.stats import multivariate_normal
import tools
import pickle

data_source_file_name = "./dataSource/data.csv"

num_classes = 51  # 51个志愿者
dataNum_eachClass = 400  # 每种样本的数据量
num_feature = 31  # 31维数据
positive_test_size = 200  # 这三行代码使得正负测试样本数量比例为1比1
negative_begin = 0
negative_end = 5
ht_index = list(range(0, 31, 3))
pp_index = list(range(1, 31, 3))
index1 = ht_index
index2 = pp_index

ht_pp_index = list(ht_index + pp_index)
ht_pp_index.sort()
index = ht_pp_index  # 多元高斯算法考虑最佳的ht_pp特征，全部扔到协方差里而非采用融合算法

all_data = np.loadtxt(data_source_file_name, dtype=float, delimiter=",", skiprows=1, usecols=tuple(range(3, 34)))

X1 = []
eer_array1_train_size = []  # 特征合并多元高斯
X11 = []
eer_array11_train_size = []  # 特征分开多元高斯
X = []
eer_array2_train_size = []  # 单元高斯
eer_array3_train_size = []  # 曼哈顿距离
for train_size in range(200, 201, 2):
    X.append(train_size)
    print("train_size:" + str(train_size))

    # HT和FT合并多元高斯
    if train_size > 21:
        X1.append(train_size)
        eer_array1_classes = []
        for i in range(num_classes):
            train_data = all_data[i * dataNum_eachClass:i * dataNum_eachClass + train_size, index]
            #  todo filter?结果会更好
            positive_test_data = all_data[
                                 i * dataNum_eachClass + train_size:i * dataNum_eachClass + train_size + positive_test_size,
                                 index]
            train_cov = np.cov(train_data, rowvar=False)  # 计算协方差矩阵
            mean = np.mean(train_data, axis=0, keepdims=True)[0]  # axis=1返回横轴平均值,0纵轴平均值[0]从矩阵表示的向量中拿出向量表示的向量
            positive_scores = multivariate_normal.pdf(positive_test_data, mean=mean, cov=train_cov).tolist()
            negative_scores = []
            for j in range(num_classes):
                if j == i:
                    continue
                negative_test_data = all_data[
                                     j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                                     index]
                temp_scores = multivariate_normal.pdf(negative_test_data, mean=mean, cov=train_cov)
                negative_scores += temp_scores.tolist()
            # negative_scores.sort()
            score_array = positive_scores + negative_scores
            label_array = [1] * len(positive_scores) + [0] * len(negative_scores)
            far, tpr, thresholds = roc_curve(label_array, score_array)
            EER = tools.calculate_eer(far, tpr, thresholds)
            eer_array1_classes.append(EER)
        eer_array1_classes = tools.select_eer(eer_array1_classes)
        eer1 = sum(eer_array1_classes) / len(eer_array1_classes)
        eer_array1_train_size.append(eer1)
        print("多元高斯(HT和FT合并)eer=" + str(eer1))

    # HT和FT分开多元高斯
    if train_size > 11:
        X11.append(train_size)
        eer_array1_classes = []
        for i in range(num_classes):
            train_data1 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + train_size,
                          index1]
            positive_test_data1 = all_data[
                                  i * dataNum_eachClass + train_size:i * dataNum_eachClass + train_size + positive_test_size,
                                  index1]
            positive_scores1 = tools.calculate_score_by_cov(train_data1, positive_test_data1)
            train_data2 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + train_size,
                          index2]
            positive_test_data2 = all_data[
                                  i * dataNum_eachClass + train_size:i * dataNum_eachClass + train_size + positive_test_size,
                                  index2]
            positive_scores2 = tools.calculate_score_by_cov(train_data2, positive_test_data2)
            positive_scores = positive_scores1 * positive_scores2  # 用乘法来衔接两种特征
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
                negative_scores += (negative_score1 * negative_score2).tolist()
            score_array = positive_scores.tolist() + negative_scores
            label_array = [1] * len(positive_scores) + [0] * len(negative_scores)
            far, tpr, thresholds = roc_curve(label_array, score_array)
            EER = tools.calculate_eer(far, tpr, thresholds)
            eer_array1_classes.append(EER)
        eer_array1_classes = tools.select_eer(eer_array1_classes)
        eer1 = sum(eer_array1_classes) / len(eer_array1_classes)
        eer_array11_train_size.append(eer1)
        print("多元高斯(HT和FT分开)eer=" + str(eer1))

    eer_array2_classes = []
    for i in range(num_classes):
        train_data1 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + train_size,
                      index1]
        positive_test_data1 = all_data[
                              i * dataNum_eachClass + train_size:i * dataNum_eachClass + train_size + positive_test_size,
                              index1]
        positive_scores1 = tools.calculate_score_by_std(train_data1, positive_test_data1)
        train_data2 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + train_size,
                      index2]
        positive_test_data2 = all_data[
                              i * dataNum_eachClass + train_size:i * dataNum_eachClass + train_size + positive_test_size,
                              index2]
        positive_scores2 = tools.calculate_score_by_std(train_data2, positive_test_data2)
        positive_scores = positive_scores1 * positive_scores2  # 用乘法来衔接两种特征
        negative_scores = []
        for j in range(num_classes):
            if j == i:
                continue
            negative_test_data1 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                                  index1]
            negative_test_data2 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                                  index2]
            negative_score1 = tools.calculate_score_by_std(train_data1, negative_test_data1)
            negative_score2 = tools.calculate_score_by_std(train_data2, negative_test_data2)
            negative_scores += (negative_score1 * negative_score2).tolist()
        score_array = positive_scores.tolist() + negative_scores
        label_array = [1] * len(positive_scores) + [0] * len(negative_scores)
        far, tpr, thresholds = roc_curve(label_array, score_array)
        EER = tools.calculate_eer(far, tpr, thresholds)
        eer_array2_classes.append(EER)
    eer_array2_classes = tools.select_eer(eer_array2_classes)
    eer2 = sum(eer_array2_classes) / len(eer_array2_classes)
    eer_array2_train_size.append(eer2)
    print("单元高斯eer=" + str(eer2))

    eer_array3_classes = []
    for i in range(num_classes):
        train_data1 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + train_size,
                      index1]
        positive_test_data1 = all_data[
                              i * dataNum_eachClass + train_size:i * dataNum_eachClass + train_size + positive_test_size,
                              index1]
        positive_scores1 = tools.calculate_score_by_Manhattan(train_data1, positive_test_data1)
        train_data2 = all_data[i * dataNum_eachClass:i * dataNum_eachClass + train_size,
                      index2]
        positive_test_data2 = all_data[
                              i * dataNum_eachClass + train_size:i * dataNum_eachClass + train_size + positive_test_size,
                              index2]
        positive_scores2 = tools.calculate_score_by_Manhattan(train_data2, positive_test_data2)
        positive_scores = -positive_scores1 * positive_scores2  # 用乘法来衔接两种特征
        negative_scores = []
        for j in range(num_classes):
            if j == i:
                continue
            negative_test_data1 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                                  index1]
            negative_test_data2 = all_data[j * dataNum_eachClass + negative_begin:j * dataNum_eachClass + negative_end,
                                  index2]
            negative_score1 = tools.calculate_score_by_Manhattan(train_data1, negative_test_data1)
            negative_score2 = tools.calculate_score_by_Manhattan(train_data2, negative_test_data2)
            negative_scores += (-negative_score1 * negative_score2).tolist()
        score_array = positive_scores.tolist() + negative_scores
        label_array = [1] * len(positive_scores) + [0] * len(negative_scores)
        far, tpr, thresholds = roc_curve(label_array, score_array)
        EER = tools.calculate_eer(far, tpr, thresholds)
        eer_array3_classes.append(EER)
    eer_array3_classes = tools.select_eer(eer_array3_classes)
    eer3 = sum(eer_array3_classes) / len(eer_array3_classes)
    eer_array3_train_size.append(eer3)
    print("曼哈顿eer=" + str(eer3))
# with open("./results/静态认证冷启动2(去掉3个最大值eer值的用户).pk", "wb") as f:
#     pickle.dump(X1, f)
#     pickle.dump(X11, f)
#     pickle.dump(X, f)
#     pickle.dump(eer_array1_train_size, f)
#     pickle.dump(eer_array11_train_size, f)
#     pickle.dump(eer_array2_train_size, f)
#     pickle.dump(eer_array3_train_size, f)

