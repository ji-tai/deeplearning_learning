#获取官方已有的数据集
import torch
import torchvision
import torch.nn.functional as F

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

# --------------------------------------------------

class LogisticRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# --------------------------------------------------

# 损失函数

# 交叉熵损失	多分类	nn.CrossEntropyLoss
# 二元交叉熵	二分类	nn.BCEWithLogitsLoss
criterion = torch.nn.BCELoss(size_average=False)
# 更新器
optimizer = torch.optim.SGD(model.parameters(), lr=0.7)

# --------------------------------------------------

for epoch in range(1000):

    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)

    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)

x_t = torch.Tensor(x).view((200, 1))

y_t = model(x_t)

y = y_t.data.numpy()

plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
