import matplotlib.pylab as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad


epoch_list = []
cost_list = []
lr_val = 0.01
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    loss_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr_val * grad_val
    print('epoch:', epoch, 'w=', w, 'loss=', loss_val)
    epoch_list.append(epoch)
    cost_list.append(loss_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
