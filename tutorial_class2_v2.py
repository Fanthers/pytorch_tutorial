import matplotlib.pylab as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


w = 1.0

def forward(x):
    return x * w


def lost(xs, ys):
    y_pred = forward(xs)
    return (y_pred - ys) ** 2
    

def gradient(xs, ys):
    return 2 * xs * (xs * w - ys)


epoch_list = []
lost_list = []
lr_val = 0.01
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = lost(x, y)
        grad_val = gradient(x, y)
        w -= lr_val * grad_val
        print('minibatch:', epoch, 'w=', w, 'loss=', loss_val)
    epoch_list.append(epoch)
    lost_list.append(loss_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,lost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

# 折中 batch/Mini-batch 一部分数据作为一个batch全部处理更新权重
