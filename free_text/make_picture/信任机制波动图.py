import matplotlib.pyplot as plt
import random

max_value = 10  # 上限值
min_value = 0  # 下限值，触及则警告
C = 1  # 最大奖励
B = 1  # 最大乘法，因为random的范围在0-1之间，所以底下代码不用涉及B和C
begin = 10  # 初始值
y = begin
x_length = 500  # 序列长度，一个序列单位包含两个按下和抬起两个击键动作
add_or_sub_threshold = 2 / 5  # 随机数（0，1）大于它则是非法用户，小于则合法

X = list(range(0, x_length))
Y = []
for i in range(0, x_length):
    add_or_sub_value = random.random()
    sign = 1
    if add_or_sub_value < add_or_sub_threshold:
        sign = 1
    else:
        sign = -1
    increment = sign * random.random();
    y += increment
    if y >= max_value:
        y = max_value
        Y.append(y)
    elif y <= min_value:
        Y.append(y)
        y = begin
    else:
        Y.append(y)
plt.plot(X, Y, color='green')
plt.plot([0, x_length], [0, 0], color='red')
# plt.legend(loc='upper left')
plt.xlabel('Keystroke event number')
plt.ylabel('Trust model score')
plt.xlim(0, x_length)
plt.ylim(min_value - 5, max_value)
# plt.title('EER of different proportion of hold time and flight time')
plt.show()
