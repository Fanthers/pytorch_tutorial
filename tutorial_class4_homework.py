import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 构造对象，并说明输入输出的维数，第三个参数默认为true，表示用到b

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(reduction='sum')
epoch_list = np.arange(0, 100, 1)
plt.figure()
optimizer_list = ['Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'sgd']
for opt_list in optimizer_list:
    if opt_list == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    elif opt_list == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif opt_list == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    elif opt_list == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)
    elif opt_list == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    elif opt_list == 'Rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
    elif opt_list == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss_list = []
    for epoch in range(100):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    plt.plot(epoch_list, loss_list)
    plt.title(opt_list)
    plt.show()

