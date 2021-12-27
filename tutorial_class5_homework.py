import math
import torch

pred = torch.tensor([[-0.2], [0.2], [0.8]])
target = torch.tensor([[0.0], [0.0], [1.0]])

sigmoid = torch.nn.Sigmoid()
pred_s = sigmoid(pred)
print(pred_s)

result = 0
i = 0
for label in target:
    if label.item() == 0:
        result += math.log(1-pred_s[i].item())
    else:
        result += math.log(pred_s[i].item())
    i += 1

result /= 3
print('bce: ', -result)
loss = torch.nn.BCELoss()
print('BCEloss:', loss(pred_s, target).item())