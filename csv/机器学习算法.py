import csv
import numpy as np
import os
import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# 预定义数据
data_source_file_name = "./dataSource/data.csv"  # todo 改成相对目录
num_classes = 51  # 51个志愿者
dataNum_eachClass = 400  # 每种样本的数据量
num_feature = 31  # 31维数据
train_ratio = 0.7
test_ratio = 1 - train_ratio

np.random.random
# 准备数据
x_data = np.loadtxt(data_source_file_name, dtype=float, delimiter=",", skiprows=1, usecols=tuple(range(3, 34)))
y_data = np.zeros((num_classes * dataNum_eachClass, 1), dtype=int)
for i in range(num_classes):
    for j in range(dataNum_eachClass):
        y_data[i * dataNum_eachClass + j, 0] = i  # 生成对应标签值
x_train = np.zeros((0, num_feature), dtype=float)
y_train = np.zeros((0, 1), dtype=int)
x_test = np.zeros((0, num_feature), dtype=float)
y_test = np.zeros((0, 1), dtype=int)
for i in range(num_classes):
    x_train = np.vstack((x_train,
                         x_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass), :]))
    x_test = np.vstack((x_test,
                        x_data[i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
                        :]))
    y_train = np.vstack((y_train,
                         y_data[i * dataNum_eachClass:i * dataNum_eachClass + int(train_ratio * dataNum_eachClass), :]))
    y_test = np.vstack((y_test,
                        y_data[i * dataNum_eachClass + int(train_ratio * dataNum_eachClass):(i + 1) * dataNum_eachClass,
                        :]))
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# 训练模型
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=num_feature))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print("end.")
