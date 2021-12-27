import numpy as np
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 4.0]
y_data = [2.0, 4.0, 8.0]


def forward(x):
    return x * w


def loss(x_val, y_val):
    y_pred = forward(x_val)
    return (y_pred - y_val) ** 2


w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print('w =', w)
    l_sum = 0
    for x, y in zip(x_data, y_data):
        y_pred_val = forward(x)
        loss_val = loss(x, y)
        l_sum += loss_val
        print('\t', x, y, y_pred_val, loss_val)
    print('MSE=', l_sum/len(x_data))
    w_list.append(w)
    mse_list.append(l_sum/len(x_data))


plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()

