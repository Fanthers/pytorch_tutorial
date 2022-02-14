import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('./diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
#  optimizer = torch.optim.SGD(data.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    loss_list.append(loss.item())
    epoch_list.append(epoch)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 999:
        # torch.where(condition, x, y) 满足条件返回x, 不满足返回y
        y_pred_label = torch.where(y_pred>0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        # torch.eq(input,output) 相同为1，不同为0
        acc = torch.eq(y_pred_label, y_data).sum().item()/y_data.size(0)
        print('loss=', loss.item(), 'acc=', acc)

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()