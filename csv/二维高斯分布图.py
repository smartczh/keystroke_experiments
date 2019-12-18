import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
import math

data_source_file_name = "./dataSource/data.csv"
all_data = np.loadtxt(data_source_file_name, dtype=float, delimiter=",", skiprows=1, usecols=tuple(range(3, 34)))

num_classes = 51  # 51个志愿者
dataNum_eachClass = 400  # 每种样本的数据量
num_feature = 31  # 31维数据
train_ratio = 1
test_ratio = 1 - train_ratio
negative_begin = 0
negative_end = 5

num = 200
l = np.linspace(-0.1, 0.4, num)
X, Y = np.meshgrid(l, l)

for i in range(num_classes):
    for j in range(num_feature):
        i = 20
        j = 19
        index = (j, j + 3)
        print(i, index)
        train_data = all_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass), index]
        train_cov = np.cov(train_data, rowvar=False)  # 计算协方差矩阵
        mean = np.mean(train_data, axis=0, keepdims=True)[0]  # axis=1返回横轴平均值,0纵轴平均值[0]从矩阵表示的向量中拿出向量表示的向量
        u = mean  # 均值
        o = train_cov  # 协方差矩阵
        pos = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2)), axis=2)  # 定义坐标点
        a = np.dot((pos - u), np.linalg.inv(o))  # o的逆矩阵
        b = np.expand_dims(pos - u, axis=3)
        # Z = np.dot(a.reshape(200*200,2),(pos-u).reshape(200*200,2).T)
        Z = np.zeros((num, num), dtype=np.float32)
        for i in range(num):
            Z[i] = [np.dot(a[i, j], b[i, j]) for j in range(num)]  # 计算指数部分
        Z = np.exp(Z * (-0.5)) / (2 * np.pi * math.sqrt(np.linalg.det(o)))
        fig = plt.figure()  # 绘制图像
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.6, cmap=cm.coolwarm)
        cset = ax.contour(X, Y, Z, 10, zdir='z', offset=0, cmap=cm.coolwarm)  # 绘制xy面投影
        cset = ax.contour(X, Y, Z, zdir='x', offset=-4, cmap=mpl.cm.winter)  # 绘制zy面投影
        cset = ax.contour(X, Y, Z, zdir='y', offset=4, cmap=mpl.cm.winter)  # 绘制zx面投影
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
