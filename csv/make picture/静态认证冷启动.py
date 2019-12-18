import pickle
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'MicroSoft Yahei',
        'weight': 'bold',
        'size': 12}

matplotlib.rc("font", **font)



with open("results/静态认证强化实验1.pk", "rb") as f:
    X1 = pickle.load(f)
    X11 = pickle.load(f)
    X = pickle.load(f)
    eer_array1_train_size = pickle.load(f)
    eer_array11_train_size = pickle.load(f)
    eer_array2_train_size = pickle.load(f)
    eer_array3_train_size = pickle.load(f)
    plt.plot(X1, eer_array1_train_size, color='black', label="合并多元高斯")
    plt.plot(X11, eer_array11_train_size, color='red', label="分开多元高斯")
    plt.plot(X, eer_array2_train_size, color='green', label="单元高斯")
    plt.plot(X, eer_array3_train_size, color='yellow', label="曼哈顿距离")
    #plt.plot(x2, eer_y2, color='red', label="Summation")
    # plt.plot(x3, eer_y3, color='blue', label="logarithm norm operation")
    # plt.plot(x4, eer_y4, color='yellow', label="sqrt operation")
    plt.legend(loc='upper right')
    plt.xlabel('proportion of HT&HTP')
    plt.ylabel('EER')
    #plt.title('EER of different proportion of hold time and flight time')
    plt.show()