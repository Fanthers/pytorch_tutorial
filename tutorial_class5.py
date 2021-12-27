import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.], [0.], [1.]])


class LogisticRessionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRessionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRessionModel()


# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for opoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(opoch, loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print('w= ', model.linear.weight.item())
print('b= ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred= ', y_test.item())