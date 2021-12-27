# 小试牛刀
# import torch
# a = torch.tensor([1.0])
# a.requires_grad = True # 或者 a.requires_grad_()
# print(a)  # tensor([1.], requires_grad=True)
# print(a.data)  # tensor([1.])
# print(a.type())  # torch.FloatTensor
# print(a.data.type())  # torch.FloatTensor
# print(a.grad)  # None
# print(type(a.grad))  # <class 'NoneType'>

import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad_()

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\tgrad", x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    print('epoch:', epoch, l.item())
print('predict (after training)', 4, forward(4).item())