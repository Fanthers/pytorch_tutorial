# y = x * w + b
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x_data = [1.0, 2.0, 4.0]
y_data = [2.0, 4.0, 8.0]


def forward(x):
    return x * w + b


def loss(x_val, y_val):
    y_pred = forward(x_val)
    return (y_pred - y_val) ** 2


w = np.arange(0.0, 2.1, 0.1)
b = np.arange(0.0, 4.1, 0.1)
[w, b] = np.meshgrid(w, b)
print(len(w))

w_list = []
mse_list = []
l_sum = 0
# flag = True
for x, y in zip(x_data, y_data):
    y_pred_val = forward(x)
    # if flag:
    #     print(y_pred_val)
    #     flag = False
    loss_val = loss(x, y)
    l_sum += loss_val
    # print('\t', x, y, y_pred_val, loss_val)
# print('MSE=', l_sum/len(x_data))
w_list.append(w)
mse_list.append(l_sum/len(x_data))


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, mse_list[0])
plt.show()

